import torch
import torch.nn.functional as F

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class FastGrowMask:
    """
    GrowMask with the same result, using OpenCV instead of SciPy's Python-level loop.

    ComfyUI's GrowMask runs one full SciPy pass per pixel of growth, which costs over a second
    per node at expand=64 on a 1536px mask and then contends with every other lane on the CPU.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "KWJ/Mask"
    DESCRIPTION = "Grows or shrinks a mask. Pixel-identical to GrowMask, computed with OpenCV."

    def execute(self, mask, expand, tapered_corners):
        if cv2 is None:
            raise RuntimeError("KWJ_FastGrowMask needs OpenCV (opencv-python).")
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        if expand == 0:
            return (mask.clone(),)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) if tapered_corners else np.ones((3, 3), np.uint8)
        grow = cv2.dilate if expand > 0 else cv2.erode
        grown = [
            torch.from_numpy(grow(item.cpu().numpy(), kernel, iterations=abs(expand), borderType=cv2.BORDER_REFLECT))
            for item in mask
        ]
        return (torch.stack(grown),)


class ScaleToMegapixels:
    """Resizes to about `megapixels`, keeping aspect ratio, with both sides snapped to multiples of 16."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 16.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "KWJ/Image"
    DESCRIPTION = "Scales an image to a pixel budget. Sides stay multiples of 16 for Flux latents."

    def execute(self, image, megapixels):
        height, width = image.shape[1], image.shape[2]
        scale = (megapixels * 1e6 / (width * height)) ** 0.5
        new_width = max(16, round(width * scale / 16) * 16)
        new_height = max(16, round(height * scale / 16) * 16)
        if (new_width, new_height) == (width, height):
            return (image,)
        mode = "area" if new_width < width else "bicubic"
        scaled = F.interpolate(image.movedim(-1, 1), size=(new_height, new_width), mode=mode)
        return (scaled.movedim(1, -1).clamp(0, 1),)
