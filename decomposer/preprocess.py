from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

MIN_DIMENSION = 64
WARN_DIMENSION = 2048


class PreprocessError(Exception):
    pass


def preprocess(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise PreprocessError(
            f"Image too small ({w}x{h}). Minimum dimension is {MIN_DIMENSION}px."
        )
    if w > WARN_DIMENSION or h > WARN_DIMENSION:
        logger.warning(
            "Image is %dx%d, will be downscaled to inference resolution", w, h
        )
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image
