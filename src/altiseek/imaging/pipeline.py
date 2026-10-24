"""The one shared resampling path (CLAUDE.md §8): area-averaging downsampling, then the camera
pipeline (optional Gaussian blur, optional JPEG re-encoding).

Detector training views, real-frame evaluation and world rendering all end in
:func:`render_view`, so every image the detector sees went through the same image model. Oblique
frames are first warped onto a finer ground grid (``altiseek.geometry.rectify``) and then reach
the target resolution through this same function.
"""

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from altiseek.config import Config


class CameraPipeline(Config):
    """Degradations applied after resampling, in output pixels."""

    #: Standard deviation of the Gaussian blur; 0 disables it.
    blur_sigma_px: float = 0.0
    #: JPEG quality (1-95) of the re-encoding; None disables it.
    jpeg_quality: int | None = None


def area_resize(image: NDArray[np.uint8], height_px: int, width_px: int) -> NDArray[np.uint8]:
    """Anti-aliased area-averaging (box-filter) resize of an ``H x W x 3`` image; no upsampling."""
    if height_px > image.shape[0] or width_px > image.shape[1]:
        raise ValueError(
            f"area_resize only downsamples: {image.shape[:2]} -> {(height_px, width_px)}"
        )
    if (height_px, width_px) == image.shape[:2]:
        return image.copy()
    resized = Image.fromarray(image).resize((width_px, height_px), Image.Resampling.BOX)
    return np.array(resized)


def camera_pipeline(image: NDArray[np.uint8], pipeline: CameraPipeline) -> NDArray[np.uint8]:
    """Blur, then JPEG round-trip, as configured. Deterministic for a given Pillow build."""
    out = Image.fromarray(image)
    if pipeline.blur_sigma_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=pipeline.blur_sigma_px))
    if pipeline.jpeg_quality is not None:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=pipeline.jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            out = decoded.convert("RGB")
    return np.array(out)


def render_view(
    image: NDArray[np.uint8], height_px: int, width_px: int, pipeline: CameraPipeline
) -> NDArray[np.uint8]:
    """The shared path: area-averaging downsample to the output size, then the camera pipeline."""
    return camera_pipeline(area_resize(image, height_px, width_px), pipeline)
