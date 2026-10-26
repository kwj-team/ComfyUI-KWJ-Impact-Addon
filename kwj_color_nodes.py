import torch
import torch.nn.functional as F


def srgb_to_oklab(image):
    linear = torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )
    r, g, b = linear.unbind(-1)
    l = torch.pow((0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).clamp(min=0), 1 / 3)
    m = torch.pow((0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).clamp(min=0), 1 / 3)
    s = torch.pow((0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).clamp(min=0), 1 / 3)
    return torch.stack(
        (
            0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
        ),
        dim=-1,
    )


def oklab_to_srgb(lab):
    L, a, b = lab.unbind(-1)
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
    linear = torch.stack(
        (
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
        ),
        dim=-1,
    ).clamp(0, 1)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear, 1 / 2.4) - 0.055,
    ).clamp(0, 1)


def ramp(x, low, high):
    return ((x - low) / (high - low)).clamp(0, 1)


def skin_likelihood(lab):
    """Warm, moderately saturated, not dark. Hair, eyes, lips and shadows score low."""
    L, a, b = lab.unbind(-1)
    chroma = torch.sqrt(a * a + b * b)
    hue = torch.rad2deg(torch.atan2(b, a))
    return (
        ramp(L, 0.4, 0.5)
        * ramp(chroma, 0.015, 0.03)
        * (1 - ramp(chroma, 0.15, 0.19))
        * ramp(hue, 5, 15)
        * (1 - ramp(hue, 70, 80))
    )


def resize_like(tensor, height, width):
    """Resizes a BHWC image or BHW mask to height x width."""
    if tensor.shape[1] == height and tensor.shape[2] == width:
        return tensor
    if tensor.dim() == 3:
        return F.interpolate(tensor.unsqueeze(1), size=(height, width), mode="bilinear").squeeze(1)
    return F.interpolate(tensor.movedim(-1, 1), size=(height, width), mode="bilinear").movedim(1, -1)


def batch_item(tensor, index):
    return tensor[min(index, tensor.shape[0] - 1)]


class MaskedSkinToneMatch:
    """
    Shifts the skin inside `mask` toward the skin tone the same region had in `image_ref`.

    Face swaps pull skin tone from the reference photo, which leaves a pale face on a warmer
    illustrated body. Feeding the pre-swap crop as `image_ref` restores the illustration's tone.
    Only the mean OKLab offset of skin pixels moves, so the new face's shading and features stay.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "mask": ("MASK",),
                "luminance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_luminance_shift": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "max_color_shift": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.005}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    CATEGORY = "KWJ/Color"
    DESCRIPTION = (
        "Matches skin tone inside the mask to the same region of a reference image "
        "(OKLab mean shift on skin-coloured pixels, capped)."
    )

    # Fewer measured skin pixels than this means there is no face to match.
    MIN_SKIN_PIXELS = 500
    # Only the solid part of a feathered mask is measured.
    MEASURE_MASK_THRESHOLD = 0.8

    def doit(
        self,
        image_ref,
        image_target,
        mask,
        luminance_strength=1.0,
        color_strength=1.0,
        max_luminance_shift=0.1,
        max_color_shift=0.04,
    ):
        height, width = image_target.shape[1], image_target.shape[2]
        ref = resize_like(image_ref[..., :3].float(), height, width)
        masks = mask.float()
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        masks = resize_like(masks, height, width)

        strength = torch.tensor([luminance_strength, color_strength, color_strength])
        limit = torch.tensor([max_luminance_shift, max_color_shift, max_color_shift])

        outputs = []
        for index in range(image_target.shape[0]):
            target = image_target[index, ..., :3].float()
            item_mask = batch_item(masks, index)
            target_lab = srgb_to_oklab(target)
            ref_lab = srgb_to_oklab(batch_item(ref, index))
            target_skin = skin_likelihood(target_lab)

            measured = item_mask >= self.MEASURE_MASK_THRESHOLD
            target_pixels = measured & (target_skin >= 0.5)
            ref_pixels = measured & (skin_likelihood(ref_lab) >= 0.5)
            if (
                int(target_pixels.sum()) < self.MIN_SKIN_PIXELS
                or int(ref_pixels.sum()) < self.MIN_SKIN_PIXELS
            ):
                outputs.append(image_target[index])
                continue

            shift = ref_lab[ref_pixels].mean(0) - target_lab[target_pixels].mean(0)
            shift = torch.maximum(torch.minimum(shift * strength.to(shift), limit.to(shift)), -limit.to(shift))

            # Pupils, lashes and nostrils stay put; the mask edge fades the shift out.
            weight = item_mask * ramp(target_lab[..., 0], 0.35, 0.6) * target_skin
            shifted = oklab_to_srgb(target_lab + shift * weight.unsqueeze(-1))
            result = torch.where(weight.unsqueeze(-1) > 0, shifted, target)
            if image_target.shape[-1] > 3:
                result = torch.cat((result, image_target[index, ..., 3:].float()), dim=-1)
            outputs.append(result.to(image_target.dtype))

        return (torch.stack(outputs),)
