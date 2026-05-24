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
