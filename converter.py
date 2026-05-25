#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {'.webp', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
JPEG_QUALITY = 95


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    background = Image.new('RGB', image.size, (255, 255, 255))
    rgba = image.convert('RGBA')
    background.paste(rgba, mask=rgba.split()[3])
    return background


def convert_file(input_path: Path) -> None:
    output_path = input_path.with_suffix('.jpg')
    original_size = input_path.stat().st_size

    with Image.open(input_path) as img:
        rgb_img = to_rgb(img)
        rgb_img.save(output_path, format='JPEG', quality=JPEG_QUALITY)

    output_size = output_path.stat().st_size
    pct_change = (output_size / original_size - 1) * 100

    print(f"{input_path.name} → {output_path.name}")
    print(f"  Quality: {JPEG_QUALITY}  |  "
          f"{original_size:,} B → {output_size:,} B  ({pct_change:+.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert images to JPEG at quality 95.'
    )
    parser.add_argument('files', nargs='+', type=Path, metavar='FILE',
                        help='Input image files (webp, png, jpeg, tiff)')
    args = parser.parse_args()

    for path in args.files:
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping {path.name}: unsupported format ({path.suffix})", file=sys.stderr)
            continue
        try:
            convert_file(path)
        except Exception as exc:
            print(f"Error converting {path.name}: {exc}", file=sys.stderr)


if __name__ == '__main__':
    main()
