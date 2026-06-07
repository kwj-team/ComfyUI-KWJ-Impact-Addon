import numpy as np
import torch


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    if len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask


class SEGSFilterClosestMask:
    methods = ["IoU", "Centroid"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "mask": ("MASK",),
                "match_method": (cls.methods,),
                "threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "return_all_scores": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SEGS", "FLOAT")
    RETURN_NAMES = ("filtered_segs", "best_score")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "doit"
    CATEGORY = "KWJ/ImpactPack/Operation"
    DESCRIPTION = (
        "Filters SEGS to find the segment with the best match to the provided mask. "
        "IoU returns the highest intersection-over-union score; Centroid returns the closest distance."
    )

    @staticmethod
    def calculate_centroid(mask_np):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (mask_np.shape[1] / 2, mask_np.shape[0] / 2)

        return (float(np.mean(x_indices)), float(np.mean(y_indices)))

    @staticmethod
    def make_full_size_seg_mask(seg, result_shape):
        seg_mask_np = np.zeros((result_shape[0], result_shape[1]), dtype=np.float32)
        x1, y1, x2, y2 = seg.crop_region

        if isinstance(seg.cropped_mask, torch.Tensor):
            cropped_mask = seg.cropped_mask.detach().cpu().numpy()
        else:
            cropped_mask = seg.cropped_mask

        if len(cropped_mask.shape) == 3:
            cropped_mask = np.max(cropped_mask, axis=0)

        crop_h, crop_w = cropped_mask.shape
        paste_y1, paste_y2 = max(y1, 0), min(y1 + crop_h, result_shape[0])
        paste_x1, paste_x2 = max(x1, 0), min(x1 + crop_w, result_shape[1])
        view_crop_h = paste_y2 - paste_y1
        view_crop_w = paste_x2 - paste_x1

        if view_crop_h > 0 and view_crop_w > 0:
            crop_y1 = paste_y1 - y1
            crop_x1 = paste_x1 - x1
            seg_mask_np[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_mask[
                crop_y1 : crop_y1 + view_crop_h,
                crop_x1 : crop_x1 + view_crop_w,
            ]

        return seg_mask_np

    def doit(self, segs, mask, match_method, threshold=0.0, return_all_scores=False):
        if not segs or len(segs) < 2 or len(segs[1]) == 0:
            return ((0, 0), []), 0.0

        mask_tensor = make_2d_mask(mask)
        mask_np = mask_tensor.detach().cpu().numpy()

        result_shape = segs[0]
        best_seg = None
        all_scores = []

        if match_method == "IoU":
            best_score = -1.0
            for seg in segs[1]:
                seg_mask_np = self.make_full_size_seg_mask(seg, result_shape)
                intersection = np.sum(np.logical_and(seg_mask_np > 0, mask_np > 0))
                union = np.sum(np.logical_or(seg_mask_np > 0, mask_np > 0))
                current_score = float(intersection / union) if union > 0 else 0.0

                all_scores.append(current_score)
                if current_score > best_score and current_score >= threshold:
                    best_score = current_score
                    best_seg = seg

            final_best_score = best_score if best_score >= 0 else 0.0
        elif match_method == "Centroid":
            best_score = float("inf")
            target_centroid = self.calculate_centroid(mask_np)
            max_possible_distance = float(np.sqrt(result_shape[0] ** 2 + result_shape[1] ** 2)) or 1.0

            for seg in segs[1]:
                seg_mask_np = self.make_full_size_seg_mask(seg, result_shape)
                seg_centroid = self.calculate_centroid(seg_mask_np)
                current_score = float(
                    np.sqrt(
                        (seg_centroid[0] - target_centroid[0]) ** 2
                        + (seg_centroid[1] - target_centroid[1]) ** 2
                    )
                )

                all_scores.append(current_score)
                normalized_distance_score = 1.0 - (current_score / max_possible_distance)
                if current_score < best_score and normalized_distance_score >= threshold:
                    best_score = current_score
                    best_seg = seg

            final_best_score = best_score if best_score != float("inf") else 0.0
        else:
            raise ValueError(f"Unknown match_method: {match_method}")

        if return_all_scores:
            print(f"[KWJ Impact Addon] Scores ({match_method}) for all segments: {all_scores}")

        result_segs = [best_seg] if best_seg is not None else []
        return (result_shape, result_segs), final_best_score
