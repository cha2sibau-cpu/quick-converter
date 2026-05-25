import pytest
from PIL import Image
from pathlib import Path

from converter import to_rgb, convert_file


class TestToRgb:
    def test_rgb_passthrough_returns_same_object(self):
        import numpy as np
        img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        result = to_rgb(img)
        assert result is img

    def test_rgba_fully_transparent_becomes_white(self):
        import numpy as np
        img = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
        result = to_rgb(img)
        assert result.mode == 'RGB'
        assert all(v == 255 for v in list(result.getdata())[0])

    def test_rgba_fully_opaque_preserves_color(self):
        import numpy as np
        img = Image.new('RGBA', (4, 4), (100, 150, 200, 255))
        result = to_rgb(img)
        arr = np.array(result)
        assert arr[0, 0, 0] == 100
        assert arr[0, 0, 1] == 150
        assert arr[0, 0, 2] == 200

    def test_la_converts_to_rgb(self):
        img = Image.new('LA', (4, 4), (128, 200))
        result = to_rgb(img)
        assert result.mode == 'RGB'
        assert result.size == (4, 4)

    def test_l_grayscale_converts_to_rgb(self):
        img = Image.new('L', (4, 4), 128)
        result = to_rgb(img)
        assert result.mode == 'RGB'
        assert result.size == (4, 4)

    def test_palette_converts_to_rgb(self):
        import numpy as np
        rgb_img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        palette_img = rgb_img.convert('P')
        result = to_rgb(palette_img)
        assert result.mode == 'RGB'


class TestConvertFile:
    def _write_png(self, path: Path, mode: str = 'RGB') -> None:
        img = Image.new('RGB', (64, 64), (128, 128, 128))
        if mode == 'RGBA':
            img = img.convert('RGBA')
        img.save(path)

    def test_png_produces_jpg_output(self, tmp_path):
        src = tmp_path / 'test.png'
        self._write_png(src)
        convert_file(src)
        assert (tmp_path / 'test.jpg').exists()

    def test_output_is_valid_jpeg(self, tmp_path):
        src = tmp_path / 'test.png'
        self._write_png(src)
        convert_file(src)
        with Image.open(tmp_path / 'test.jpg') as img:
            assert img.format == 'JPEG'

    def test_rgba_converts_without_error(self, tmp_path):
        src = tmp_path / 'test.png'
        self._write_png(src, mode='RGBA')
        convert_file(src)
        assert (tmp_path / 'test.jpg').exists()

    def test_jpeg_extension_normalised_to_jpg(self, tmp_path):
        src = tmp_path / 'test.jpeg'
        self._write_png(src)
        convert_file(src)
        assert (tmp_path / 'test.jpg').exists()

    def test_output_size_is_positive(self, tmp_path):
        src = tmp_path / 'test.png'
        self._write_png(src)
        convert_file(src)
        assert (tmp_path / 'test.jpg').stat().st_size > 0

    def test_output_dimensions_match_input(self, tmp_path):
        src = tmp_path / 'test.png'
        img = Image.new('RGB', (800, 600), (100, 150, 200))
        img.save(src)
        convert_file(src)
        with Image.open(tmp_path / 'test.jpg') as out:
            assert out.size == (800, 600)
