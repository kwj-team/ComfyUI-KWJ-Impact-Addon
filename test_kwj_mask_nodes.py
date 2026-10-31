import unittest

try:
    import cv2
    import numpy as np
    import scipy.ndimage
    import torch

    HAS_DEPS = hasattr(torch, "tensor")
except ImportError:
    HAS_DEPS = False

if HAS_DEPS:
    from kwj_mask_nodes import FastGrowMask, ScaleToMegapixels


def comfy_grow_mask(mask, expand, tapered_corners):
    """ComfyUI's GrowMask, kept here as the reference for parity."""
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    output = mask.numpy()
    for _ in range(abs(expand)):
        step = scipy.ndimage.grey_erosion if expand < 0 else scipy.ndimage.grey_dilation
        output = step(output, footprint=kernel)
    return output


@unittest.skipUnless(HAS_DEPS, "torch, numpy, scipy or OpenCV is not installed")
class FastGrowMaskTests(unittest.TestCase):
    def setUp(self):
        # Soft edges, and shapes that touch the image border, where kernels disagree first.
        mask = np.zeros((256, 320), np.float32)
        cv2.circle(mask, (40, 2), 30, 1.0, -1)
        cv2.ellipse(mask, (180, 130), (50, 70), 20, 0, 360, 1.0, -1)
        cv2.circle(mask, (316, 250), 18, 0.6, -1)
        self.mask = torch.from_numpy(cv2.GaussianBlur(mask, (0, 0), 2))
        self.node = FastGrowMask()

    def test_matches_comfy_grow_mask(self):
        for tapered in (True, False):
            for expand in (1, 2, 16, 64, -1, -20):
                with self.subTest(tapered=tapered, expand=expand):
                    (out,) = self.node.execute(self.mask.unsqueeze(0), expand, tapered)
                    self.assertTrue(np.array_equal(out[0].numpy(), comfy_grow_mask(self.mask, expand, tapered)))

    def test_handles_batches(self):
        batch = torch.stack((self.mask, self.mask.flip(0)))
        (out,) = self.node.execute(batch, 8, True)
        self.assertEqual(out.shape, batch.shape)
        self.assertTrue(np.array_equal(out[1].numpy(), comfy_grow_mask(self.mask.flip(0), 8, True)))

    def test_expand_zero_returns_a_copy(self):
        batch = self.mask.unsqueeze(0)
        (out,) = self.node.execute(batch, 0, True)
        self.assertTrue(torch.equal(out, batch))
        out[0, 0, 0] = 0.5
        self.assertNotEqual(float(batch[0, 0, 0]), 0.5)


@unittest.skipUnless(HAS_DEPS, "torch, numpy, scipy or OpenCV is not installed")
class ScaleToMegapixelsTests(unittest.TestCase):
    def test_sides_are_multiples_of_16_near_the_budget(self):
        (out,) = ScaleToMegapixels().execute(torch.rand(1, 512, 512, 3), 0.5)
        self.assertEqual(out.shape, (1, 704, 704, 3))
        (out,) = ScaleToMegapixels().execute(torch.rand(1, 1000, 1500, 3), 0.5)
        self.assertEqual(out.shape[1] % 16, 0)
        self.assertEqual(out.shape[2] % 16, 0)
        self.assertLess(abs(out.shape[1] * out.shape[2] / 0.5e6 - 1), 0.1)

    def test_unchanged_size_returns_the_input(self):
        image = torch.rand(1, 704, 704, 3)
        (out,) = ScaleToMegapixels().execute(image, 0.4956)
        self.assertIs(out, image)


if __name__ == "__main__":
    unittest.main()
