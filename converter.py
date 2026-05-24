#!/usr/bin/env python3
import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

SUPPORTED_EXTENSIONS = {'.webp', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
MIN_QUALITY = 70
MAX_QUALITY = 95
SSIM_THRESHOLD = 0.999


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    background = Image.new('RGB', image.size, (255, 255, 255))
    rgba = image.convert('RGBA')
    background.paste(rgba, mask=rgba.split()[3])
    return background


def find_optimal_quality(rgb_array: np.ndarray) -> tuple[int, float, bytes]:
    """Return (quality, ssim_score, jpeg_bytes) at lowest quality with SSIM >= SSIM_THRESHOLD."""
    best_quality = None
    best_ssim = None
    best_bytes = None
    last_score = None
    last_bytes = None

    for quality in range(MAX_QUALITY, MIN_QUALITY - 1, -1):
        buf = io.BytesIO()
        Image.fromarray(rgb_array).save(buf, format='JPEG', quality=quality)
        jpeg_bytes = buf.getvalue()

        buf.seek(0)
        compressed = np.array(Image.open(buf).convert('RGB'))
        score = structural_similarity(rgb_array, compressed, channel_axis=-1, data_range=255)

        last_score = score
        last_bytes = jpeg_bytes

        if score >= SSIM_THRESHOLD:
            best_quality = quality
            best_ssim = score
            best_bytes = jpeg_bytes
        else:
            break

    if best_bytes is not None:
        return best_quality, best_ssim, best_bytes

    # Fallback: even quality 95 fell below threshold — use it with a warning
    print(f"  Warning: SSIM {last_score:.6f} < {SSIM_THRESHOLD} even at quality {MAX_QUALITY}",
          file=sys.stderr)
    return MAX_QUALITY, last_score, last_bytes


def convert_file(input_path: Path) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
