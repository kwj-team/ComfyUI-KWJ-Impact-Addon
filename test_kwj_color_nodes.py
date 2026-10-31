import unittest

try:
    import torch

    HAS_TORCH = hasattr(torch, "tensor")
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from kwj_color_nodes import MaskedSkinToneMatch, oklab_to_srgb, ramp, skin_likelihood, srgb_to_oklab


def rgb(r, g, b):
    return torch.tensor([r, g, b], dtype=torch.float32) / 255


def scene(face_rgb, size=64, face=40):
    """A blue background with a square skin patch and a dark 'pupil' in it."""
    image = rgb(60, 80, 140).repeat(1, size, size, 1)
    image[0, :face, :face] = face_rgb
    image[0, 10:13, 10:13] = rgb(30, 25, 25)
    mask = torch.zeros(1, size, size)
    mask[0, :face, :face] = 1.0
    return image, mask


def full_image_match(image_ref, image_target, mask, node, strength=(1.0, 1.0, 1.0), limit=(0.1, 0.04, 0.04)):
    """The whole-image calculation the bounding-box version must match exactly."""
    target = image_target[0, ..., :3].float()
    ref = image_ref[0, ..., :3].float()
    item_mask = mask[0].float()
    target_lab = srgb_to_oklab(target)
    ref_lab = srgb_to_oklab(ref)
    target_skin = skin_likelihood(target_lab)
    measured = item_mask >= node.MEASURE_MASK_THRESHOLD
    target_pixels = measured & (target_skin >= 0.5)
    ref_pixels = measured & (skin_likelihood(ref_lab) >= 0.5)
    if int(target_pixels.sum()) < node.MIN_SKIN_PIXELS or int(ref_pixels.sum()) < node.MIN_SKIN_PIXELS:
        return image_target
    strength = torch.tensor(strength)
    limit = torch.tensor(limit)
    shift = ref_lab[ref_pixels].mean(0) - target_lab[target_pixels].mean(0)
    shift = torch.maximum(torch.minimum(shift * strength.to(shift), limit.to(shift)), -limit.to(shift))
    weight = item_mask * ramp(target_lab[..., 0], 0.35, 0.6) * target_skin
    shifted = oklab_to_srgb(target_lab + shift * weight.unsqueeze(-1))
    return torch.where(weight.unsqueeze(-1) > 0, shifted, target).unsqueeze(0)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MaskedSkinToneMatchTests(unittest.TestCase):
    def setUp(self):
        self.node = MaskedSkinToneMatch()
        self.node.MIN_SKIN_PIXELS = 100

    def test_oklab_round_trip(self):
        image = torch.rand(1, 8, 8, 3)
        self.assertLess(float((oklab_to_srgb(srgb_to_oklab(image)) - image).abs().max()), 1e-4)

    def test_pale_face_moves_to_reference_tone(self):
        ref, mask = scene(rgb(225, 170, 140))
        pale, _ = scene(rgb(240, 205, 195))
        (out,) = self.node.doit(ref, pale, mask, max_luminance_shift=0.5, max_color_shift=0.2)

        ref_lab = srgb_to_oklab(ref[0, 20, 20])
        before = (srgb_to_oklab(pale[0, 20, 20]) - ref_lab).abs().sum()
        after = (srgb_to_oklab(out[0, 20, 20]) - ref_lab).abs().sum()
        self.assertLess(float(after), float(before) * 0.1)

    def test_pixels_outside_mask_and_dark_pixels_are_untouched(self):
        ref, mask = scene(rgb(225, 170, 140))
        pale, _ = scene(rgb(240, 205, 195))
        (out,) = self.node.doit(ref, pale, mask)

        self.assertTrue(torch.equal(out[0, 50:, 50:], pale[0, 50:, 50:]))
        self.assertTrue(torch.equal(out[0, 10:13, 10:13], pale[0, 10:13, 10:13]))

    def test_shift_is_capped(self):
        ref, mask = scene(rgb(150, 95, 70))
        pale, _ = scene(rgb(240, 205, 195))
        (out,) = self.node.doit(ref, pale, mask, max_luminance_shift=0.05, max_color_shift=0.01)

        delta = srgb_to_oklab(out[0, 20, 20]) - srgb_to_oklab(pale[0, 20, 20])
        self.assertLessEqual(abs(float(delta[0])), 0.05 + 1e-4)
        self.assertLessEqual(abs(float(delta[1])), 0.01 + 1e-4)
        self.assertLessEqual(abs(float(delta[2])), 0.01 + 1e-4)

    def test_too_few_skin_pixels_returns_target_unchanged(self):
        ref, mask = scene(rgb(225, 170, 140), face=6)
        pale, _ = scene(rgb(240, 205, 195), face=6)
        (out,) = self.node.doit(ref, pale, mask)

        self.assertTrue(torch.equal(out, pale))

    def test_reference_and_mask_at_another_resolution(self):
        ref, mask = scene(rgb(225, 170, 140), size=128, face=80)
        pale, _ = scene(rgb(240, 205, 195))
        (out,) = self.node.doit(ref, pale, mask)

        self.assertEqual(tuple(out.shape), tuple(pale.shape))
        self.assertFalse(torch.equal(out[0, 20, 20], pale[0, 20, 20]))

    def test_matches_the_whole_image_calculation(self):
        torch.manual_seed(7)
        for name, (top, bottom, left, right) in {
            "centre": (20, 90, 30, 110),
            "touches_corner": (0, 50, 0, 60),
            "touches_far_edge": (70, 120, 100, 160),
        }.items():
            with self.subTest(name):
                image = rgb(230, 175, 150).repeat(1, 120, 160, 1) * (0.85 + 0.15 * torch.rand(1, 120, 160, 1))
                ref = rgb(205, 145, 115).repeat(1, 120, 160, 1) * (0.85 + 0.15 * torch.rand(1, 120, 160, 1))
                mask = torch.zeros(1, 120, 160)
                mask[0, top:bottom, left:right] = 1.0
                mask[0, top, left:right] = 0.4
                (out,) = self.node.doit(ref, image, mask)
                self.assertFalse(torch.equal(out, image))
                expected = full_image_match(ref, image, mask, self.node)
                # SIMD rounding of pow() depends on array offset, so floats can differ by a few ULP.
                self.assertLess(float((out - expected).abs().max()), 1e-6)
                self.assertTrue(torch.equal((out * 255).round(), (expected * 255).round()))

    def test_empty_mask_returns_the_target(self):
        ref, _ = scene(rgb(225, 170, 140))
        pale, _ = scene(rgb(240, 205, 195))
        (out,) = self.node.doit(ref, pale, torch.zeros(1, 64, 64))
        self.assertTrue(torch.equal(out, pale))

    def test_too_few_skin_pixels_returns_the_target(self):
        ref, _ = scene(rgb(225, 170, 140))
        pale, _ = scene(rgb(240, 205, 195))
        mask = torch.zeros(1, 64, 64)
        mask[0, 20:24, 20:24] = 1.0
        (out,) = self.node.doit(ref, pale, mask)
        self.assertTrue(torch.equal(out, pale))


if __name__ == "__main__":
    unittest.main()
