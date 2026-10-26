import unittest

try:
    import torch

    HAS_TORCH = hasattr(torch, "tensor")
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from kwj_color_nodes import MaskedSkinToneMatch, oklab_to_srgb, srgb_to_oklab


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


if __name__ == "__main__":
    unittest.main()
