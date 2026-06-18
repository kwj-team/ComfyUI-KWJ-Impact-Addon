"""
@title: KWJ Impact Pack Addon
@nickname: KWJ Impact Addon
@description: Small addon nodes that depend on ComfyUI-Impact-Pack without replacing it.
"""

import os
import sys


def _ensure_impact_pack_on_path():
    """Find a separately installed Impact Pack when it has not loaded yet."""
    try:
        import impact.core  # noqa: F401
        return
    except Exception:
        pass

    addon_dir = os.path.dirname(os.path.realpath(__file__))
    custom_nodes_dir = os.path.dirname(addon_dir)

    for entry in os.listdir(custom_nodes_dir):
        candidate_dir = os.path.join(custom_nodes_dir, entry)
        modules_dir = os.path.join(candidate_dir, "modules")
        impact_core = os.path.join(modules_dir, "impact", "core.py")

        if candidate_dir == addon_dir:
            continue

        if os.path.isfile(impact_core) and modules_dir not in sys.path:
            sys.path.append(modules_dir)
            return


_ensure_impact_pack_on_path()

from .kwj_impact_nodes import SEGSFilterClosestMask
from .kwj_url_loader import KWJ_CachedImageLoadFromURL


NODE_CLASS_MAPPINGS = {
    "KWJ_SEGSFilterClosestMask": SEGSFilterClosestMask,
    "KWJ_CachedImageLoadFromURL": KWJ_CachedImageLoadFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KWJ_SEGSFilterClosestMask": "SEGS Filter (closest mask)",
    "KWJ_CachedImageLoadFromURL": "KWJ Cached Image Load From URL",
}


def _node_id_is_available(node_id):
    try:
        import nodes

        return node_id not in nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return True


if _node_id_is_available("SEGSFilterClosestMask"):
    NODE_CLASS_MAPPINGS["SEGSFilterClosestMask"] = SEGSFilterClosestMask
    NODE_DISPLAY_NAME_MAPPINGS["SEGSFilterClosestMask"] = "SEGS Filter (closest mask, legacy)"


try:
    import cm_global

    cm_global.register_extension(
        "ComfyUI-KWJ-Impact-Addon",
        {
            "version": "0.1.1",
            "name": "KWJ Impact Pack Addon",
            "nodes": set(NODE_CLASS_MAPPINGS.keys()),
            "description": "Addon nodes that depend on a separately installed ComfyUI-Impact-Pack.",
        },
    )
except Exception:
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
