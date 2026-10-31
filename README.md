# KWJ Impact Pack Addon

This is a small ComfyUI custom node addon that depends on
[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) being installed separately.

It does not register itself as Impact Pack and does not load Impact Pack's node table. Install the official
Impact Pack normally, then install this folder as a separate custom node directory.

## Nodes

- `KWJ_SEGSFilterClosestMask` / `SEGS Filter (closest mask)`
- `SEGSFilterClosestMask` / `SEGS Filter (closest mask, legacy)` when that node ID is not already registered
- `KWJ_MaskedSkinToneMatch` / `KWJ Masked Skin Tone Match` (`KWJ/Color`): shifts skin inside a mask to the tone
  the same region had in a reference image. Used after face swaps so the new face matches the illustrated body.
  Needs only torch, not Impact Pack.
- `KWJ_CachedImageLoadFromURL` / `KWJ Cached Image Load From URL` (`KWJ/Loaders`), plus the unprefixed
  `Cached Image Load From URL` alias when that node ID is free.
- `KWJ_KleinEditComposite` / `Klein Edit Composite` (`KWJ/Klein`) and
  `KWJ_KleinEditCompositeFast` / `Klein Edit Composite (Optimized)`, with `KleinEditComposite` and
  `KleinEditCompositeFast` legacy aliases when those IDs are free. See below. These need OpenCV, not Impact Pack.

## Klein Edit Composite

Composites an AI-edited image back onto its original, detecting *only what actually changed* so the original
background survives everywhere it was not meaningfully altered. The pipeline aligns the pair with two-pass SIFT
plus MAGSAC homography, catches leftover motion with DIS optical flow, builds a hybrid difference map from LAB
Delta E and Sobel gradients, cleans the mask up (opening-by-reconstruction, hole filling, island pruning, border
bleed, grow/shrink), optionally applies Reinhard colour transfer measured on background pixels only, and finally
feathers the mask to alpha-blend the result. Outputs are `composited_image`, `change_mask`, a `report` of the
parameters and statistics, and an optional `debug_gallery` of the intermediate steps.

`KWJ_KleinEditCompositeFast` runs that same pipeline with the same controls, plus a `performance_mode` selector.
It is not an approximation of the original by default — it removes duplicated work. SIFT and CLAHE run once per
image instead of once per alignment pass, the original-image half of the difference map is built once instead of
four times, pass 3 reuses pass 1 when pass 2 finds no new homography, LAB conversion goes through
`cv2.cvtColor`, and `poisson_blend_edges` solves at reduced resolution (the clone membrane is harmonic, so it
has no high frequencies to lose — 42–46 dB PSNR for 45–170× less time). Measured on a 1024² edit pair, best of
3 on a 10-core CPU:

| Settings | `KWJ_KleinEditComposite` | `KWJ_KleinEditCompositeFast` (`balanced`) | Speedup |
| --- | --- | --- | --- |
| 1024², defaults | 1.77s | 0.63s | 2.8× |
| 2048², defaults | 6.93s | 2.36s | 2.9× |
| 1024², `poisson_blend_edges` on | 14.33s | 0.90s | 16× |

| `performance_mode` | Feature detection | Optical flow | Poisson solve | Reproducible |
| --- | --- | --- | --- | --- |
| `exact` | full resolution, 10000 features | full resolution | full resolution | no (keeps FLANN) |
| `balanced` *(default)* | ≤1600px, 4000 features | full resolution | ≤768px | yes |
| `fast` | ≤1024px, 2500 features | half resolution | ≤512px | yes |

The optimized node also fixes the original's non-determinism: the original samples its auto-threshold with an
unseeded `np.random.choice` and matches descriptors with FLANN's randomized kd-forest, so identical runs give
masks differing by a few percent IoU. `balanced` and `fast` are bit-reproducible. Use `exact` to confirm the
optimized node reproduces the original's masks before switching a workflow over; it introduces no
approximations. `balanced` and `fast` fit the homography at reduced resolution and lift it back, shifting mask
edges slightly — by less, in practice, than the original node's own run-to-run variation.

Everything here runs on the CPU, because SIFT, optical flow and Poisson blending have no GPU path in the OpenCV
builds ComfyUI installs.

## Installation

1. Install the official `ComfyUI-Impact-Pack` in `ComfyUI/custom_nodes`.
2. Install this addon in a different folder name, for example:

   ```bash
   cd ComfyUI/custom_nodes
   git clone <this-repo-url> ComfyUI-KWJ-Impact-Addon
   ```

3. Restart ComfyUI.

The nodes appear under `KWJ/ImpactPack/Operation`, `KWJ/Color`, `KWJ/Loaders` and `KWJ/Klein`.

## Notes

- The primary node types are prefixed with `KWJ_` to avoid collisions with Impact Pack and other packs.
- Workflows saved with an unprefixed type such as `SEGSFilterClosestMask` or `KleinEditComposite` can keep
  loading through the legacy alias when no other installed node has already registered that ID.
- If another pack also registers one of those unprefixed IDs, whichever extension ComfyUI loads for it will be
  used. Use the `KWJ_`-prefixed type in new workflows when you need this addon's implementation. In particular,
  the Klein aliases stand down if the original `comfyui-klein-edit-composite` is still installed alongside this
  addon.
- This addon imports `impact.core` from the separately installed Impact Pack. If Impact Pack is missing,
  ComfyUI will report that dependency at startup. Only the SEGS node needs it; the colour, loader and Klein
  nodes work without it.
- The Klein nodes need OpenCV, which ComfyUI core does not install itself (Impact Pack already pulls it in).
  Only main-module APIs are used, so the `contrib` build is not required.
- The Klein nodes came from [`supermansundies/comfyui-klein-edit-composite`](https://github.com/supermansundies/comfyui-klein-edit-composite);
  `KWJ_KleinEditCompositeFast` is a local optimization of that node.
