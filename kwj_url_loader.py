"""
KWJ URL image loader — flat file cache, no metadata.json index.

Safe for SimplePods that share one volume: each pod uses a distinct filename
prefix (env or hostname) and downloads are written atomically (*.part → replace).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError


def _pod_cache_prefix() -> str:
    for key in ("KWJ_POD_ID", "SIMPLEPOD_POD_ID", "POD_ID"):
        value = os.environ.get(key, "").strip()
        if value:
            return re.sub(r"[^a-zA-Z0-9_-]+", "-", value)

    host = os.environ.get("HOSTNAME", "").strip()
    if not host:
        return "pod"

    host = host.split(":")[0]
    runpod_match = re.match(r"^([a-z0-9]+)-8188\.proxy\.runpod\.net$", host, re.I)
    if runpod_match:
        return runpod_match.group(1)

    cf_match = re.match(r"^([a-z0-9-]+)\.trycloudflare\.com$", host, re.I)
    if cf_match:
        return cf_match.group(1)

    generic_match = re.match(r"^([a-z0-9-]+)\.", host, re.I)
    if generic_match:
        return generic_match.group(1)

    return re.sub(r"[^a-zA-Z0-9_-]+", "-", host)[:32] or "pod"


def _cache_dir() -> Path:
    root = Path(folder_paths.get_input_directory()) / ".kwj_url_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(url: str) -> Path:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    prefix = _pod_cache_prefix()
    return _cache_dir() / f"{prefix}-{digest}.webp"


def _request_headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": "KWJ-ComfyUI-URL-Loader/1.0"}
    if "alien.childbook.ai" in url:
        headers["Authorization"] = "Basic dXNlcjpwYXNzd29yZA=="
    return headers


def _download_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_request_headers(url))
    retry_delays = (1, 2, 3)

    for attempt in range(len(retry_delays) + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data:
                raise ValueError(f"Empty response from URL: {url}")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code != 502 or attempt >= len(retry_delays):
                raise
            time.sleep(retry_delays[attempt])


def _save_webp(path: Path, image: Image.Image) -> None:
    tmp_path = path.with_suffix(path.suffix + ".part")
    image.save(tmp_path, format="WEBP", quality=92, method=6)
    os.replace(tmp_path, path)


def _validate_image_bytes(data: bytes) -> None:
    if not data:
        raise ValueError("Empty image data")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Invalid image dimensions")


def _try_read_cache(path: Path) -> bytes | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    data = path.read_bytes()
    try:
        _validate_image_bytes(data)
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError):
        path.unlink(missing_ok=True)
        return None
    return data


def _fetch_and_cache(url: str, keep_alpha_channel: bool) -> bytes:
    path = _cache_path(url)
    raw = _download_url(url)
    try:
        _validate_image_bytes(raw)
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"URL did not return a valid image: {url}") from exc

    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)

    if keep_alpha_channel and image.mode in ("RGBA", "LA"):
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    _save_webp(path, image)
    cached = path.read_bytes()
    try:
        _validate_image_bytes(cached)
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError):
        path.unlink(missing_ok=True)
        raise ValueError(f"Failed to cache a valid image from URL: {url}")
    return cached


def _load_bytes(url: str, keep_alpha_channel: bool) -> bytes:
    path = _cache_path(url)
    cached = _try_read_cache(path)
    if cached is not None:
        return cached
    return _fetch_and_cache(url, keep_alpha_channel)


def _tensor_from_bytes(data: bytes, keep_alpha_channel: bool) -> torch.Tensor:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)

    if keep_alpha_channel and image.mode in ("RGBA", "LA"):
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    output_images = []
    for frame in ImageSequence.Iterator(image):
        frame = ImageOps.exif_transpose(frame)
        if keep_alpha_channel and frame.mode in ("RGBA", "LA"):
            frame = frame.convert("RGBA")
        else:
            frame = frame.convert("RGB")
        tensor = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        output_images.append(tensor.unsqueeze(0))

    if len(output_images) > 1:
        return torch.cat(output_images, dim=0)
    return output_images[0]


class KWJ_CachedImageLoadFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "keep_alpha_channel": ("BOOLEAN", {"default": False}),
                "output_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "KWJ/Loaders"
    DESCRIPTION = (
        "Downloads an image from a URL into a per-pod flat cache (no metadata.json). "
        "Safe for shared SimplePod volumes."
    )

    def load(self, url, keep_alpha_channel=False, output_mode=False):
        if not url or not str(url).startswith("http"):
            raise ValueError(f"Invalid image URL: {url}")

        if output_mode:
            raise ValueError("KWJ_CachedImageLoadFromURL does not support output_mode=True")

        data = _load_bytes(str(url), keep_alpha_channel)
        return (_tensor_from_bytes(data, keep_alpha_channel),)
