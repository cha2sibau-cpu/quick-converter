import numpy as np
import pytest
from PIL import Image

from converter import find_optimal_quality, to_rgb


class TestToRgb:
    def test_rgb_passthrough_returns_same_object(self):
        img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        result = to_rgb(img)
        assert result is img

    def test_rgba_fully_transparent_becomes_white(self):
        img = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
        result = to_rgb(img)
        assert result.mode == 'RGB'
        assert np.all(np.array(result) == 255)

    def test_rgba_fully_opaque_preserves_color(self):
        img = Image.new('RGBA', (4, 4), (100, 150, 200, 255))
        result = to_rgb(img)
        arr = np.array(result)
        assert np.all(arr[:, :, 0] == 100)
        assert np.all(arr[:, :, 1] == 150)
        assert np.all(arr[:, :, 2] == 200)

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
        rgb_img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        palette_img = rgb_img.convert('P')
        result = to_rgb(palette_img)
        assert result.mode == 'RGB'


def _make_rgb_array(complexity: str = 'moderate') -> np.ndarray:
    rng = np.random.default_rng(42)
    if complexity == 'flat':
        return np.full((64, 64, 3), 128, dtype=np.uint8)
    elif complexity == 'moderate':
        # 256x256 gentle gradient (1 value/pixel) — low-frequency content JPEG compresses cleanly
        r = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        g = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 256))
        b = np.full((256, 256), 128, dtype=np.uint8)
        return np.stack([r, g, b], axis=-1)
    elif complexity == 'noise':
        return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    raise ValueError(f"Unknown complexity: {complexity}")


class TestFindOptimalQuality:
    def test_returns_quality_in_valid_range(self):
        arr = _make_rgb_array('moderate')
        quality, _, _ = find_optimal_quality(arr)
        assert 70 <= quality <= 95

    def test_ssim_at_or_above_threshold_when_achievable(self):
        # Flat image: SSIM >= 0.999 is achievable, so the result must honour the threshold
        arr = _make_rgb_array('flat')
        _, ssim_score, _ = find_optimal_quality(arr)
        assert ssim_score >= 0.999

    def test_returns_valid_jpeg_bytes(self):
        arr = _make_rgb_array('moderate')
        _, _, jpeg_bytes = find_optimal_quality(arr)
        assert isinstance(jpeg_bytes, bytes)
        assert jpeg_bytes[:2] == b'\xff\xd8'

    def test_flat_image_uses_minimum_quality(self):
        arr = _make_rgb_array('flat')
        quality, ssim_score, _ = find_optimal_quality(arr)
        assert quality == 70
        assert ssim_score >= 0.999

    def test_noise_image_still_returns_bytes(self):
        arr = _make_rgb_array('noise')
        quality, ssim_score, jpeg_bytes = find_optimal_quality(arr)
        assert 70 <= quality <= 95
        assert len(jpeg_bytes) > 0


from pathlib import Path
from converter import convert_file


class TestConvertFile:
    def _write_png(self, path: Path, mode: str = 'RGB') -> None:
        arr = _make_rgb_array('flat')
        img = Image.fromarray(arr)
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
