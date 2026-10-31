"""
Klein Edit Composite (Optimized)
================================

Same pipeline and same controls as ``KleinEditComposite``, with the redundant work
removed.  The original node is left untouched so the two can be compared side by
side in the same workflow.

Everything here still runs on the CPU — SIFT, FLANN, DIS optical flow and
``seamlessClone`` have no GPU path in the OpenCV wheels ComfyUI installs.  The
speedup comes from not doing the same work several times:

  1. SIFT/CLAHE run once per image instead of once per alignment pass.  OpenCV
     applies ``retainBest`` before filtering keypoints by the detection mask, so
     masking the keypoint list after the fact is equivalent to passing the mask in.
  2. The original-image side of the difference map (blur, LAB, gradient magnitude)
     is identical across all four ``_compute_diff_map`` calls, so it is computed once.
  3. When pass 2 fails to produce a new homography the old code re-derived pass 1's
     flow, warp and difference maps from the very same array.  Those results are
     now reused directly.
  4. LAB conversion goes through ``cv2.cvtColor``, which uses the same D65 sRGB
     transform as the hand-rolled version but avoids two general-exponent ``pow``
     passes per call (``np.where`` evaluated both branches for every pixel).
  5. The ``np.mgrid`` remap grid is cached rather than rebuilt — as int64 — on
     every warp.
  6. Descriptor matching uses brute force when the feature budget is small enough
     for it to beat FLANN, which it is in the two faster presets.
  7. The MAD threshold sampler draws from a seeded generator with replacement
     instead of permuting the whole array.
  8. ``poisson_blend_edges`` solves at reduced resolution.  This is by far the
     largest single win when the option is on: ``seamlessClone`` over a full-frame
     mask at 1024px takes about eight seconds, roughly twelve times the cost of
     everything else in the node put together.

The last two also make the node reproducible.  The original has two unseeded
sources of randomness — ``np.random.choice`` in the MAD threshold sampler, and
FLANN's kd-forest, which picks split dimensions from the C ``rand()`` stream — so
running it twice on identical inputs gives masks differing by a couple of percent
IoU.  MAGSAC, despite being a RANSAC variant, is deterministic here.

``performance_mode`` additionally trades a little accuracy for speed by running
feature detection and optical flow at reduced resolution; see the tooltip.
"""

import numpy as np
import cv2
from collections import namedtuple

from .kwj_klein_edit_composite import (
    FLOW_PRESETS,
    KleinEditComposite,
    _apply_heatmap,
    _bleed_mask,
    _blur_kernel_for_diag,
    _create_side_by_side,
    _diag,
    _draw_sift_matches,
    _fill_holes,
    _flow_to_color,
    _grow_mask,
    _lab_to_rgb,
    _local_threshold_map,
    _open_mask,
)


# ---------------------------------------------------------------------------
# Performance presets
# ---------------------------------------------------------------------------
#
# sift_max_dim  : longest side the grayscale image is downscaled to before SIFT.
#                 The homography is fitted in that reduced space and lifted back
#                 to full resolution.  SIFT cost is roughly linear in pixel count.
# sift_features : MAGSAC needs a few hundred solid correspondences, not ten thousand,
#                 and a smaller budget also brings matching under _BF_MAX_DESCRIPTORS.
# flow_scale    : resolution factor for DIS optical flow.  The flow field is median
#                 blurred immediately afterwards anyway, so fine detail is discarded
#                 either way.
# poisson_max_dim: longest side the Poisson solve runs at (0 = full resolution).
#                 seamlessClone scales worse than linearly — a full-frame mask at
#                 1024px takes ~8s, which dwarfs the whole rest of the pipeline.

PERF_PRESETS = {
    "exact": {
        "name": "exact",
        "sift_max_dim": 0,
        "sift_features": 10000,
        "sift_contrast": 0.03,
        "flann_checks": 80,
        "flow_scale": 1.0,
        "poisson_max_dim": 0,
    },
    "balanced": {
        "name": "balanced",
        "sift_max_dim": 1600,
        "sift_features": 4000,
        "sift_contrast": 0.04,
        "flann_checks": 32,
        "flow_scale": 1.0,
        "poisson_max_dim": 768,
    },
    "fast": {
        "name": "fast",
        "sift_max_dim": 1024,
        "sift_features": 2500,
        "sift_contrast": 0.04,
        "flann_checks": 32,
        "flow_scale": 0.5,
        "poisson_max_dim": 512,
    },
}

_FLANN_INDEX_KDTREE = 1
_BF_MAX_DESCRIPTORS = 6000
_DFT_SMOOTH_LIMIT = 31
_LAB_WEIGHTS = np.array([0.5, 1.2, 1.2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

_MAD_SAMPLE_SIZE = 50000
_MAD_SAMPLE_SEED = 0x4B4C4E


def _auto_threshold_mad_fast(diff_map: np.ndarray, valid_mask: np.ndarray = None,
                             k: float = 6.0, min_t: float = 3.0, max_t: float = 60.0) -> float:
    """Robust MAD threshold, seeded so repeated runs agree.

    The original draws its subsample with ``np.random.choice(..., replace=False)``,
    which is unseeded and permutes the whole array.  That made the node's output
    non-reproducible — back-to-back runs on identical inputs land anywhere in a
    ~0.35 band of threshold, which is enough to flip pixels along the mask edge.
    Sampling with replacement from a fixed seed is deterministic, statistically
    equivalent at these sizes, and about six times cheaper.
    """
    if valid_mask is not None and valid_mask.sum() > 100:
        sample = diff_map[valid_mask > 0.5]
    else:
        sample = diff_map.reshape(-1)

    if sample.size > _MAD_SAMPLE_SIZE:
        idx = np.random.default_rng(_MAD_SAMPLE_SEED).integers(0, sample.size, _MAD_SAMPLE_SIZE)
        sample = sample[idx]

    med = float(np.median(sample))
    mad = max(float(np.median(np.abs(sample - med))), 0.5)  # enforce a noise floor

    return float(np.clip(med + k * mad, min_t, max_t))


# ---------------------------------------------------------------------------
# Fast colour conversion
# ---------------------------------------------------------------------------

def _rgb_to_lab_fast(rgb: np.ndarray) -> np.ndarray:
    """D65 sRGB to LAB via OpenCV. Input must be float32 in [0, 1]."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


# ---------------------------------------------------------------------------
# Difference map with the reference side precomputed
# ---------------------------------------------------------------------------

_DiffRef = namedtuple("_DiffRef", ["lab", "mag"])


def _gradient_magnitude(blur: np.ndarray) -> np.ndarray:
    """Sobel magnitude of the luma channel.

    The 8-bit round-trip is deliberate: it reproduces the original node's
    quantization. ``struct_diff`` scales gradients by 40, so dropping it shifts the
    difference map by enough to flip pixels sitting near the detection threshold.
    """
    gray = cv2.cvtColor((blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _prep_diff_ref(img: np.ndarray, blur_kernel: tuple) -> _DiffRef:
    """Precompute the parts of the difference map that depend only on the original."""
    blur = cv2.GaussianBlur(img, blur_kernel, 0)
    return _DiffRef(lab=_rgb_to_lab_fast(blur), mag=_gradient_magnitude(blur))


def _diff_map_fast(ref: _DiffRef, gen_np: np.ndarray, blur_kernel: tuple) -> np.ndarray:
    """Hybrid Delta-E + gradient difference against a precomputed reference."""
    g_blur = cv2.GaussianBlur(gen_np, blur_kernel, 0)

    d = ref.lab - _rgb_to_lab_fast(g_blur)
    d *= _LAB_WEIGHTS  # damp global L shifts, boost a/b hue changes
    color_diff = np.sqrt((d * d).sum(axis=-1))

    struct_diff = np.abs(ref.mag - _gradient_magnitude(g_blur)) * 40.0

    return color_diff + struct_diff


def _apply_color_match_fast(orig_rgb: np.ndarray, gen_rgb: np.ndarray,
                            composite_mask: np.ndarray, valid_mask: np.ndarray,
                            blend_strength: float) -> tuple:
    """Reinhard transfer in LAB, using background pixels only for the statistics."""
    if blend_strength <= 0.0:
        return gen_rgb, False

    bg_mask = (composite_mask < 0.05) & (valid_mask > 0.5)
    if bg_mask.sum() < 100:
        return gen_rgb, False

    orig_lab = _rgb_to_lab_fast(orig_rgb)
    gen_lab = _rgb_to_lab_fast(gen_rgb)

    orig_mean = orig_lab[bg_mask].mean(axis=0)
    orig_std = orig_lab[bg_mask].std(axis=0) + 1e-5
    gen_mean = gen_lab[bg_mask].mean(axis=0)
    gen_std = gen_lab[bg_mask].std(axis=0) + 1e-5

    matched_lab = ((gen_lab - gen_mean) / gen_std) * orig_std + orig_mean

    # The Reinhard shift can push values outside the LAB gamut, so the inverse stays
    # on the NumPy implementation, which clamps safely. It only runs once per image.
    matched_rgb = _lab_to_rgb(matched_lab)

    blended = (matched_rgb * blend_strength) + (gen_rgb * (1.0 - blend_strength))
    return np.clip(blended, 0.0, 1.0), True


# ---------------------------------------------------------------------------
# Cached remap grid
# ---------------------------------------------------------------------------

_GRID_CACHE = {}
# A grid pair is 8 bytes per pixel, so 2048x2048 holds 33 MB. Two resolutions is
# enough for a workflow that alternates sizes without pinning much memory.
_GRID_CACHE_LIMIT = 2


def _grid(H: int, W: int) -> tuple:
    """Cached (yy, xx) float32 coordinate grid. Callers must not mutate the result."""
    key = (H, W)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        yy, xx = np.mgrid[0:H, 0:W]
        grid = (yy.astype(np.float32), xx.astype(np.float32))
        if len(_GRID_CACHE) >= _GRID_CACHE_LIMIT:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = grid
    return grid


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = _grid(H, W)
    return cv2.remap(image, xx + flow[..., 0], yy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _fwd_bwd_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = _grid(H, W)
    map_x = xx + flow_fwd[..., 0]
    map_y = yy + flow_fwd[..., 1]
    bwd_x = cv2.remap(flow_bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bwd_y = cv2.remap(flow_bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.sqrt((flow_fwd[..., 0] + bwd_x) ** 2 + (flow_fwd[..., 1] + bwd_y) ** 2)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    return (_fwd_bwd_error(flow_fwd, flow_bwd) > threshold).astype(np.float32)


def _largest_prime_factor(n: int) -> int:
    f = 2
    while f * f <= n:
        while n % f == 0:
            n //= f
        f += 1
    return n


def _poisson_roi(mask: np.ndarray) -> tuple:
    """Binary Poisson mask plus its bounding box, with the image border cleared.

    seamlessClone crashes if the mask touches the border, and it derives the clone
    centre from exactly this bounding box — computing the centre with numpy min/max
    instead introduces a one-pixel shift.
    """
    binary = (mask > 0.1).astype(np.uint8) * 255
    binary[0, :] = 0
    binary[-1, :] = 0
    binary[:, 0] = 0
    binary[:, -1] = 0
    return binary, cv2.boundingRect(binary)


def _choose_solve_size(mask: np.ndarray, max_dim: int) -> int:
    """Pick a solve resolution whose Poisson ROI transforms quickly.

    seamlessClone solves with a discrete sine transform implemented as a DFT of
    length ``2 * (n - 1)`` along each ROI dimension.  When that length has a large
    prime factor OpenCV drops to a quadratic transform, and the solve can take an
    order of magnitude longer for no benefit — a 766x692 ROI costs 2.3s where
    638x576 costs 0.18s.  Trying a few nearby sizes and keeping the smoothest is far
    cheaper than paying that penalty.
    """
    H, W = mask.shape[:2]
    longest = max(H, W)
    best_size, best_score = max_dim, None

    for size in range(max_dim, int(max_dim * 0.75), -16):
        scale = size / float(longest)
        small = cv2.resize(mask, (max(64, int(round(W * scale))), max(64, int(round(H * scale)))),
                           interpolation=cv2.INTER_NEAREST)
        _, (_, _, w, h) = _poisson_roi(small)
        if w < 8 or h < 8:
            continue

        score = max(_largest_prime_factor(2 * (w - 1)), _largest_prime_factor(2 * (h - 1)))
        if best_score is None or score < best_score:
            best_size, best_score = size, score
        if best_score <= _DFT_SMOOTH_LIMIT:
            break

    return best_size


def _seamless_blend_fast(orig_float: np.ndarray, gen_float: np.ndarray,
                         mask_float: np.ndarray, max_dim: int = 0) -> np.ndarray:
    """Poisson blend, optionally solved at reduced resolution.

    ``NORMAL_CLONE`` keeps the source gradients and adds a membrane f satisfying
    Laplace(f) = 0 inside the mask with f = orig - gen on the boundary.  A harmonic
    function carries no high frequencies, so recovering it as ``clone - gen`` at low
    resolution and upsampling costs almost nothing in accuracy — but ``seamlessClone``
    itself scales worse than linearly, so it saves a great deal of time.  At
    ``max_dim = 0`` this reduces to the full-resolution solve exactly.
    """
    H, W = orig_float.shape[:2]
    m3 = mask_float[..., np.newaxis]
    alpha_blend = np.clip(orig_float * (1.0 - m3) + gen_float * m3, 0, 1)

    scale = 1.0
    if max_dim > 0 and max(H, W) > max_dim:
        scale = _choose_solve_size(mask_float, max_dim) / float(max(H, W))

    if scale != 1.0:
        sw, sh = max(64, int(round(W * scale))), max(64, int(round(H * scale)))
        orig_s = cv2.resize(orig_float, (sw, sh), interpolation=cv2.INTER_AREA)
        gen_s = cv2.resize(gen_float, (sw, sh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask_float, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        orig_s, gen_s, mask_s = orig_float, gen_float, mask_float

    binary_mask, (x, y, w, h) = _poisson_roi(mask_s)
    if w == 0 or h == 0:
        return alpha_blend

    try:
        cloned_u8 = cv2.seamlessClone(
            (np.clip(gen_s, 0, 1) * 255).astype(np.uint8),
            (np.clip(orig_s, 0, 1) * 255).astype(np.uint8),
            binary_mask, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE,
        )
    except Exception:
        return alpha_blend

    membrane = cloned_u8.astype(np.float32) / 255.0 - gen_s
    if scale != 1.0:
        membrane = cv2.resize(membrane, (W, H), interpolation=cv2.INTER_LINEAR)

    cloned_float = np.clip(gen_float + membrane, 0, 1)
    return np.clip(orig_float * (1.0 - m3) + cloned_float * m3, 0, 1)


def _feather(sharp_mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Distance-based smoothstep falloff outward from the mask boundary."""
    if feather_px <= 0:
        return sharp_mask

    inv_mask = (sharp_mask < 0.5).astype(np.uint8)
    if inv_mask.min() != 0:
        return sharp_mask

    dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    t = np.clip(1.0 - (dist / (feather_px * 2.5)), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def _dis_flow_fast(gray_a: np.ndarray, gray_b: np.ndarray, preset: int, scale: float) -> np.ndarray:
    """DIS flow, optionally computed at reduced resolution and scaled back up."""
    h, w = gray_a.shape[:2]

    if scale < 1.0:
        sw = max(32, int(round(w * scale)))
        sh = max(32, int(round(h * scale)))
        a = cv2.resize(gray_a, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(gray_b, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh = w, h
        a, b = gray_a, gray_b

    flow = cv2.DISOpticalFlow_create(preset).calc(a, b, None)

    fx, fy = cv2.split(flow)
    flow = cv2.merge([cv2.medianBlur(fx, 5), cv2.medianBlur(fy, 5)])

    if (sw, sh) != (w, h):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        # Displacements scale with resolution, so the vectors need rescaling too.
        flow[..., 0] *= w / float(sw)
        flow[..., 1] *= h / float(sh)

    return flow


# ---------------------------------------------------------------------------
# SIFT: extract once, match many
# ---------------------------------------------------------------------------

_Features = namedtuple("_Features", ["eq", "kp", "des", "scale"])


def _extract_features(gray: np.ndarray, perf: dict) -> _Features:
    """CLAHE + SIFT on the full image. Detection masks are applied afterwards."""
    h, w = gray.shape[:2]
    scale = 1.0
    max_dim = perf["sift_max_dim"]

    if max_dim > 0 and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        gray = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    sift = cv2.SIFT_create(nfeatures=perf["sift_features"], contrastThreshold=perf["sift_contrast"])
    kp, des = sift.detectAndCompute(eq, None)

    return _Features(eq=eq, kp=list(kp) if kp else [], des=des, scale=scale)


def _filter_features(feat: _Features, mask_full: np.ndarray) -> tuple:
    """Restrict keypoints to a detection mask.

    OpenCV's SIFT retains the best ``nfeatures`` keypoints and only then discards
    those whose centre pixel falls outside the mask, so filtering here produces the
    same set as passing the mask to ``detectAndCompute`` — without re-detecting.
    ``runByPixelsMask`` truncates the coordinates, so ``int()`` matches it exactly.
    """
    if mask_full is None or feat.des is None or not feat.kp:
        return feat.kp, feat.des

    h, w = feat.eq.shape[:2]
    mask = mask_full
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    keep = [i for i, p in enumerate(feat.kp)
            if 0 <= int(p.pt[0]) < w and 0 <= int(p.pt[1]) < h and mask[int(p.pt[1]), int(p.pt[0])]]

    if not keep:
        return [], None
    return [feat.kp[i] for i in keep], feat.des[keep]


def _knn_match(des1: np.ndarray, des2: np.ndarray, checks: int) -> list:
    """Two nearest neighbours per query descriptor.

    Below a few thousand descriptors brute force beats FLANN outright — it is both
    quicker and exact, where FLANN's approximate second neighbour sometimes lets a
    bad pair through the ratio test.  It is also reproducible: FLANN's kd-forest
    picks split dimensions from an unseeded RNG, so it returns a slightly different
    match set on every call.  Above the cutoff the approximate search wins on speed.
    """
    if max(len(des1), len(des2)) <= _BF_MAX_DESCRIPTORS:
        return cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=_FLANN_INDEX_KDTREE, trees=5),
        dict(checks=checks),
    )
    return flann.knnMatch(des1, des2, k=2)


def _align_from_features(feat_orig: _Features, feat_gen: _Features, gen_float: np.ndarray,
                         out_shape: tuple, perf: dict, orig_mask: np.ndarray = None,
                         debug: bool = False) -> tuple:
    """Fit a homography from precomputed features and warp gen -> orig at full resolution."""
    H, W = out_shape
    debug_dict = {} if debug else None
    fallback_valid = np.ones((H, W), dtype=np.float32)

    kp1, des1 = _filter_features(feat_orig, orig_mask)
    kp2, des2 = feat_gen.kp, feat_gen.des

    if debug:
        debug_dict['orig_kp'] = len(kp1)
        debug_dict['gen_kp'] = len(kp2)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(feat_orig.eq, feat_gen.eq, kp1, kp2, [])
        return gen_float, None, 0, fallback_valid, debug_dict

    raw_matches = _knn_match(des1, des2, perf["flann_checks"])

    good = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if debug:
        debug_dict['matches_before_ransac'] = _draw_sift_matches(feat_orig.eq, feat_gen.eq, kp1, kp2, good, None)

    if len(good) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(feat_orig.eq, feat_gen.eq, kp1, kp2, good, None)
        return gen_float, None, 0, fallback_valid, debug_dict

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_gen = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # The correspondences live in downscaled coordinates, so the reprojection
    # tolerance is scaled to stay equivalent to 4px at full resolution.
    scale = feat_gen.scale
    ransac_method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    H_mat, inlier_mask = cv2.findHomography(pts_gen, pts_orig, ransac_method, 4.0 * scale)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if debug:
        debug_dict['match_viz'] = _draw_sift_matches(feat_orig.eq, feat_gen.eq, kp1, kp2, good, inlier_mask)
        debug_dict['n_matches'] = len(good)
        debug_dict['n_inliers'] = n_inliers

    if H_mat is None:
        return gen_float, None, 0, fallback_valid, debug_dict

    if scale != 1.0:
        # p_small = S * p_full, so H_full = S^-1 * H_small * S. The upper-left 2x2
        # block is unchanged by this, which keeps the determinant check below valid.
        S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        S_inv = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_mat = S_inv @ H_mat @ S

    det = H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0]
    if not (0.2 < det < 5.0):
        if debug:
            debug_dict['failure_reason'] = f"Bad determinant: {det:.2f} (needs 0.2-5.0)"
        return gen_float, None, 0, fallback_valid, debug_dict

    aligned = cv2.warpPerspective(
        gen_float, H_mat, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    valid_mask = cv2.warpPerspective(
        np.ones((H, W), dtype=np.float32), H_mat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if debug:
        debug_dict['homography'] = H_mat
        debug_dict['determinant'] = det

    return aligned, H_mat, n_inliers, valid_mask, debug_dict


# ---------------------------------------------------------------------------
# Core composite
# ---------------------------------------------------------------------------

def _composite_fast_impl(original_np: np.ndarray,
                         generated_np: np.ndarray,
                         delta_e_threshold: float,
                         flow_preset: int,
                         occlusion_threshold: float,
                         grow_px: int,
                         close_radius: int,
                         feather_px: float,
                         color_match_blend: float,
                         noise_removal_px: int = 0,
                         max_islands: int = 0,
                         fill_holes: bool = False,
                         use_occlusion: bool = False,
                         fill_borders: bool = True,
                         custom_mask: np.ndarray = None,
                         custom_mask_mode: str = "replace",
                         poisson_blend_edges: bool = False,
                         debug: bool = False,
                         perf: dict = None) -> tuple:

    perf = perf or PERF_PRESETS["balanced"]
    threads = cv2.getNumThreads()

    # cv2.cvtColor is undefined outside [0, 1], and the caller does not guarantee range.
    original_np = np.ascontiguousarray(np.clip(original_np, 0.0, 1.0), dtype=np.float32)
    generated_np = np.ascontiguousarray(np.clip(generated_np, 0.0, 1.0), dtype=np.float32)

    H, W = original_np.shape[:2]
    diag = _diag(H, W)
    debug_images = {} if debug else None

    gray_orig = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor((generated_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    blur_kernel = _blur_kernel_for_diag(diag)
    sk = max(blur_kernel[0], 5)
    if sk % 2 == 0:
        sk += 1

    flow_scale = perf["flow_scale"]

    auto_report = {
        "perf_mode": perf["name"],
        "cv2_threads": threads,
    }

    # Detected once, reused by every alignment pass.
    feat_orig = _extract_features(gray_orig, perf)
    feat_gen = _extract_features(gray_gen, perf)
    auto_report["sift_scale"] = feat_orig.scale
    auto_report["sift_kp"] = (len(feat_orig.kp), len(feat_gen.kp))

    # The original-image side of every difference map is identical, so build it once.
    diff_ref = _prep_diff_ref(original_np, blur_kernel)

    # ------------------------------------------------------------------
    # Custom mask override (replace mode): bypass internal detection entirely
    # ------------------------------------------------------------------
    if custom_mask is not None and custom_mask_mode == "replace":
        if debug:
            debug_images['custom_mask_override'] = np.stack([custom_mask] * 3, axis=-1)
            debug_images['mask_overlay'] = original_np.copy()
            debug_images['mask_overlay'][custom_mask > 0.5] = (
                debug_images['mask_overlay'][custom_mask > 0.5] * 0.5 + np.array([0, 0.5, 0])
            )

        gen_pre, _, inliers, valid, debug_sift = _align_from_features(
            feat_orig, feat_gen, generated_np, (H, W), perf, orig_mask=None, debug=debug
        )
        if debug:
            debug_images['pass1_sift_matches'] = debug_sift.get('match_viz', np.zeros((H, W, 3)))
            debug_images['pass1_validity_mask'] = valid.copy()

        sharp_mask = np.clip(custom_mask, 0.0, 1.0) * valid

        if fill_borders:
            sharp_mask = _bleed_mask(sharp_mask, valid)

        composite_mask = _feather(sharp_mask, feather_px)

        if not fill_borders:
            composite_mask = composite_mask * valid

        gen_pre, color_matched = _apply_color_match_fast(
            original_np, gen_pre, composite_mask, valid, color_match_blend
        )
        if color_matched:
            auto_report['color_match_applied'] = True

        if poisson_blend_edges:
            result = _seamless_blend_fast(original_np, gen_pre, composite_mask,
                                          perf["poisson_max_dim"])
        else:
            m3 = composite_mask[..., np.newaxis]
            result = np.clip(original_np * (1.0 - m3) + gen_pre * m3, 0, 1)

        gray_gen_pre = cv2.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow_fwd_final = _dis_flow_fast(gray_orig, gray_gen_pre, flow_preset, flow_scale)

        if use_occlusion:
            flow_bwd_final = _dis_flow_fast(gray_gen_pre, gray_orig, flow_preset, flow_scale)
            occ_thresh = occlusion_threshold if occlusion_threshold >= 0 else _auto_threshold_mad_fast(
                _fwd_bwd_error(flow_fwd_final, flow_bwd_final), valid, k=5.0, min_t=1.0, max_t=30.0)
            if occlusion_threshold < 0:
                auto_report['auto_occlusion'] = occ_thresh
        else:
            flow_bwd_final = None
            occ_thresh = 0

        flow_mag = np.sqrt((flow_fwd_final ** 2).sum(axis=2))
        stats = {
            "changed_pct":   100 * float((sharp_mask > 0.5).sum()) / (H * W),
            "occluded_px":   int((_fwd_bwd_error(flow_fwd_final, flow_bwd_final) > occ_thresh).sum()) if use_occlusion else 0,
            "flow_mean_px":  float(flow_mag.mean()),
            "flow_p99_px":   float(np.percentile(flow_mag, 99)),
            "median_de":     0.0,
            "resolution":    f"{W}x{H}",
            "diagonal_px":   round(diag),
            "pass1_inliers": inliers,
            "pass2_used":    False,
            "custom_mask":   True,
            "poisson_used":  poisson_blend_edges,
        }
        stats.update(auto_report)

        if debug:
            debug_images['final_flow'] = _flow_to_color(flow_fwd_final)
            debug_images['final_alignment'] = _create_side_by_side(
                original_np, gen_pre, ("Original", "Aligned (custom mask)")
            )
            debug_images['composite_breakdown'] = np.hstack([
                original_np, gen_pre, result, np.stack([composite_mask] * 3, axis=-1),
            ])

        return result, composite_mask, stats, debug_images

    # ------------------------------------------------------------------
    # PASS 1: Blind Alignment -> Coarse Mask
    # ------------------------------------------------------------------
    gen_pre_1, H_sift1, inliers_1, valid_1, debug_sift1 = _align_from_features(
        feat_orig, feat_gen, generated_np, (H, W), perf, orig_mask=None, debug=debug
    )

    if debug:
        debug_images['pass1_sift_matches'] = debug_sift1.get('match_viz', np.zeros((H, W, 3)))
        debug_images['pass1_validity_mask'] = valid_1.copy()
        valid_overlay = original_np.copy()
        valid_overlay[valid_1 < 0.5] = [1, 0, 0]
        debug_images['pass1_validity_overlay'] = valid_overlay

    gray_gen_pre_1 = cv2.cvtColor((np.clip(gen_pre_1, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow_fwd_1 = _dis_flow_fast(gray_orig, gray_gen_pre_1, flow_preset, flow_scale)
    flow_bwd_1 = _dis_flow_fast(gray_gen_pre_1, gray_orig, flow_preset, flow_scale) if use_occlusion else None

    warped_gen_1 = _warp(gen_pre_1, flow_fwd_1)

    diff_1_direct = _diff_map_fast(diff_ref, gen_pre_1, blur_kernel)
    diff_1_warped = _diff_map_fast(diff_ref, warped_gen_1, blur_kernel)

    de_thresh = delta_e_threshold if delta_e_threshold >= 0 else _auto_threshold_mad_fast(diff_1_direct, valid_1)

    blend_w_1 = np.clip(diff_1_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_1_raw = blend_w_1 * diff_1_direct + (1.0 - blend_w_1) * diff_1_warped
    delta_e_1 = cv2.GaussianBlur(delta_e_1_raw, (sk, sk), 0)

    local_thresh_1 = _local_threshold_map(delta_e_1, valid_1, diag)
    thresh_map_1 = np.minimum(de_thresh, local_thresh_1)
    coarse_mask = (delta_e_1 > thresh_map_1).astype(np.float32)

    occ_thresh = 0
    if use_occlusion:
        occ_thresh = occlusion_threshold if occlusion_threshold >= 0 else _auto_threshold_mad_fast(
            _fwd_bwd_error(flow_fwd_1, flow_bwd_1), valid_1, k=5.0, min_t=1.0, max_t=30.0)
        coarse_mask = np.maximum(coarse_mask, _occlusion_mask(flow_fwd_1, flow_bwd_1, occ_thresh))

    coarse_mask *= valid_1

    if debug:
        de_normalized = np.clip(delta_e_1 / (de_thresh * 1.5 + 1e-6), 0, 1)
        debug_images['pass1_delta_e'] = _apply_heatmap(original_np, de_normalized)
        debug_images['pass1_coarse_mask'] = coarse_mask.copy()
        debug_images['pass1_flow'] = _flow_to_color(flow_fwd_1)
        debug_images['pass1_alignment'] = _create_side_by_side(original_np, warped_gen_1, ("Original", "Pass1 Warped"))

    # ------------------------------------------------------------------
    # PASS 2: Masked SIFT Alignment (Background Only)
    # ------------------------------------------------------------------
    bg_mask_u8 = (coarse_mask < 0.1).astype(np.uint8) * 255
    safe_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg_mask_safe = cv2.erode(bg_mask_u8, safe_k)

    pass2_used = False
    final_aligned_gen, valid_2 = gen_pre_1, valid_1
    auto_report["pass2_inliers"] = 0

    if (bg_mask_safe > 0).sum() > (H * W * 0.05):
        # Only the FLANN match and the homography fit are redone here; the SIFT
        # descriptors are the ones already computed above.
        cand_gen, H_sift2, inliers_2, cand_valid, debug_sift2 = _align_from_features(
            feat_orig, feat_gen, generated_np, (H, W), perf, orig_mask=bg_mask_safe, debug=debug
        )
        if H_sift2 is not None:
            pass2_used = True
            final_aligned_gen, valid_2 = cand_gen, cand_valid
            auto_report["pass2_inliers"] = inliers_2
            if debug:
                debug_images['pass2_sift_matches'] = debug_sift2.get('match_viz', np.zeros((H, W, 3)))
                debug_images['pass2_validity_mask'] = cand_valid.copy()
    elif debug:
        debug_images['pass2_skip_reason'] = f"Background too small ({(bg_mask_safe > 0).sum()} px)"

    # ------------------------------------------------------------------
    # PASS 3: Final Precision Mask
    # ------------------------------------------------------------------
    if pass2_used:
        gray_gen_pre_2 = cv2.cvtColor((np.clip(final_aligned_gen, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        flow_fwd_2 = _dis_flow_fast(gray_orig, gray_gen_pre_2, flow_preset, flow_scale)
        flow_bwd_2 = _dis_flow_fast(gray_gen_pre_2, gray_orig, flow_preset, flow_scale) if use_occlusion else None

        warped_gen_final = _warp(final_aligned_gen, flow_fwd_2)

        diff_2_direct = _diff_map_fast(diff_ref, final_aligned_gen, blur_kernel)
        diff_2_warped = _diff_map_fast(diff_ref, warped_gen_final, blur_kernel)

        blend_w_2 = np.clip(diff_2_direct / (de_thresh + 1e-6), 0.0, 1.0)
        delta_e_2_raw = blend_w_2 * diff_2_direct + (1.0 - blend_w_2) * diff_2_warped
        delta_e_2 = cv2.GaussianBlur(delta_e_2_raw, (sk, sk), 0)

        local_thresh_2 = _local_threshold_map(delta_e_2, valid_2, diag)
        thresh_map_2 = np.minimum(de_thresh, local_thresh_2)
        sharp_mask = (delta_e_2 > thresh_map_2).astype(np.float32)

        if use_occlusion:
            sharp_mask = np.maximum(sharp_mask, _occlusion_mask(flow_fwd_2, flow_bwd_2, occ_thresh))

        sharp_mask *= valid_2
    else:
        # final_aligned_gen is gen_pre_1 itself, so pass 1's flow, warp and difference
        # maps are already the exact answer — coarse_mask included.
        flow_fwd_2, flow_bwd_2 = flow_fwd_1, flow_bwd_1
        warped_gen_final = warped_gen_1
        delta_e_2 = delta_e_1
        sharp_mask = coarse_mask.copy()
        auto_report["pass3_reused"] = True

    if delta_e_threshold < 0:
        auto_report["auto_delta_e"] = de_thresh
    if use_occlusion and occlusion_threshold < 0:
        auto_report["auto_occlusion"] = occ_thresh

    # Morphological clean up
    if noise_removal_px > 0:
        sharp_mask = _open_mask(sharp_mask, noise_removal_px)
        sharp_mask *= valid_2
    if grow_px != 0:
        sharp_mask = _grow_mask(sharp_mask, grow_px)
        sharp_mask *= valid_2
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        sharp_mask = cv2.morphologyEx(sharp_mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if max_islands > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((sharp_mask > 0.5).astype(np.uint8), connectivity=8)
        if n - 1 > max_islands:
            areas = [(stats_cc[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
            areas.sort(reverse=True)
            keep = {i for _, i in areas[:max_islands]}
            sharp_mask[np.isin(labeled, list(keep), invert=True) & (labeled > 0)] = 0

    if fill_holes:
        sharp_mask = _fill_holes(sharp_mask)
        sharp_mask *= valid_2

    # ------------------------------------------------------------------
    # Custom mask combination (add / subtract modes)
    # ------------------------------------------------------------------
    if custom_mask is not None and custom_mask_mode in ("add", "subtract"):
        cm = np.clip(custom_mask, 0.0, 1.0)
        if custom_mask_mode == "add":
            sharp_mask = np.clip(sharp_mask + cm, 0.0, 1.0)
        else:
            sharp_mask = np.clip(sharp_mask - cm, 0.0, 1.0)
        sharp_mask *= valid_2
        if debug:
            debug_images['custom_mask_override'] = np.stack([cm] * 3, axis=-1)
            auto_report['custom_mask_mode'] = custom_mask_mode

    if fill_borders:
        sharp_mask = _bleed_mask(sharp_mask, valid_2)

    composite_mask = _feather(sharp_mask, feather_px)

    if not fill_borders:
        composite_mask = composite_mask * valid_2

    final_aligned_gen, color_matched = _apply_color_match_fast(
        original_np, final_aligned_gen, composite_mask, valid_2, color_match_blend
    )
    if color_matched:
        auto_report['color_match_applied'] = True

    if poisson_blend_edges:
        result = _seamless_blend_fast(original_np, final_aligned_gen, composite_mask,
                                      perf["poisson_max_dim"])
    else:
        m3 = composite_mask[..., np.newaxis]
        result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    flow_mag = np.sqrt((flow_fwd_2 ** 2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":   100 * n_changed / (H * W),
        "occluded_px":   int((_fwd_bwd_error(flow_fwd_2, flow_bwd_2) > occ_thresh).sum()) if use_occlusion else 0,
        "flow_mean_px":  float(flow_mag.mean()),
        "flow_p99_px":   float(np.percentile(flow_mag, 99)),
        "median_de":     float(np.median(delta_e_2)),
        "resolution":    f"{W}x{H}",
        "diagonal_px":   round(diag),
        "pass1_inliers": inliers_1,
        "pass2_used":    pass2_used,
        "poisson_used":  poisson_blend_edges,
    }
    stats.update(auto_report)

    if debug:
        debug_images['final_flow'] = _flow_to_color(flow_fwd_2)
        debug_images['final_flow_magnitude'] = flow_mag / (flow_mag.max() + 1e-6)

        de_norm = np.clip(delta_e_2 / (de_thresh * 1.5 + 1e-6), 0, 1)
        debug_images['final_delta_e'] = _apply_heatmap(original_np, de_norm)

        debug_images['final_sharp_mask'] = sharp_mask.copy()
        debug_images['final_composite_mask'] = composite_mask.copy()

        mask_overlay = original_np.copy()
        mask_overlay[sharp_mask > 0.5] = (
            mask_overlay[sharp_mask > 0.5] * 0.5 + np.array([0, 0.5, 0])
        )
        debug_images['mask_overlay'] = mask_overlay

        debug_images['final_alignment'] = _create_side_by_side(
            original_np, warped_gen_final, ("Original", "Final Warped")
        )
        debug_images['composite_breakdown'] = np.hstack([
            original_np, final_aligned_gen, result, np.stack([composite_mask] * 3, axis=-1),
        ])

    return result, composite_mask, stats, debug_images


def _composite_fast(*args, **kwargs) -> tuple:
    """Run the pipeline with OpenCV's thread pool enabled for the duration.

    ``basicsr``, ``albumentations`` and several upscaler nodes call
    ``cv2.setNumThreads(0)`` at import time, which silently serialises every OpenCV
    call in the process.  Rather than change that globally and surprise other nodes,
    threading is switched on for this call only and switched back afterwards.

    ``setNumThreads`` is not symmetric: several builds only distinguish 0 (disabled)
    from any positive value, so the previous state is restored with 0 rather than
    with the thread count, and re-enabled via -1 (the build's own default) rather
    than a count of our choosing.
    """
    if cv2.getNumThreads() > 1:
        return _composite_fast_impl(*args, **kwargs)

    cv2.setNumThreads(-1)
    try:
        return _composite_fast_impl(*args, **kwargs)
    finally:
        cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_PERF_TOOLTIP = (
    "Speed/accuracy tradeoff for the alignment and optical flow stages.\n"
    "'balanced' (default): feature detection runs at up to 1600px on the long side "
    "with a trimmed keypoint budget, optical flow stays at full resolution, and the "
    "Poisson solve runs at up to 768px. Alignment quality is effectively unchanged "
    "for typical edits.\n"
    "'fast': feature detection at up to 1024px, optical flow at half resolution and "
    "the Poisson solve at up to 512px. Noticeably quicker on large images; the "
    "residual flow correction gets slightly coarser, which can soften mask edges on "
    "fine detail.\n"
    "'exact': keeps the original node's full-resolution detection and 10000-feature "
    "budget. Still faster than the original node because the duplicated work is gone, "
    "but nothing is approximated. Use this to verify the optimized node matches.\n"
    "'balanced' and 'fast' are also reproducible: repeat runs on the same inputs give "
    "the same mask. 'exact' keeps the original's randomized FLANN matching, so like "
    "the original node its mask shifts slightly from run to run."
)


class KleinEditCompositeFast(KleinEditComposite):
    """Optimized variant of KleinEditComposite.

    Identical controls plus a performance_mode selector. The pipeline is the same;
    duplicated computation has been removed and the colour conversions now use
    OpenCV's vectorized path.
    """

    CATEGORY = "KWJ/Klein"

    @classmethod
    def INPUT_TYPES(cls):
        spec = KleinEditComposite.INPUT_TYPES()
        required = dict(spec["required"])
        required["performance_mode"] = (["balanced", "fast", "exact"], {
            "default": "balanced",
            "tooltip": _PERF_TOOLTIP,
        })
        return {"required": required, "optional": dict(spec.get("optional", {}))}

    def _composite_impl(self, orig_np, gen_np, **kwargs):
        result = _composite_fast(orig_np, gen_np, perf=self._perf, **kwargs)
        self._last_stats = result[2]
        return result

    def run(self, *args, performance_mode="balanced", **kwargs):
        self._perf = PERF_PRESETS.get(performance_mode, PERF_PRESETS["balanced"])
        self._last_stats = {}

        image, mask, report, gallery = super().run(*args, **kwargs)

        stats = self._last_stats
        extra = [
            "",
            "=== Optimized pipeline ===",
            f"Performance mode: {stats.get('perf_mode', performance_mode)}",
            f"OpenCV threads:   {stats.get('cv2_threads', '?')}",
        ]
        scale = stats.get("sift_scale", 1.0)
        kp = stats.get("sift_kp")
        if kp:
            extra.append(
                f"SIFT detection:   {kp[0]}/{kp[1]} keypoints at {scale * 100:.0f}% scale (computed once)"
            )
        if stats.get("pass3_reused"):
            extra.append("Pass 3:           reused pass 1 (alignment unchanged)")

        return (image, mask, report + "\n" + "\n".join(extra), gallery)
