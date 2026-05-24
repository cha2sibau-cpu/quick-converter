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
    raise NotImplementedError


def find_optimal_quality(rgb_array: np.ndarray) -> tuple[int, float, bytes]:
    raise NotImplementedError


def convert_file(input_path: Path) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
