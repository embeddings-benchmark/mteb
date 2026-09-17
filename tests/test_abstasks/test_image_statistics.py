"""Tests for black/white image statistics and quality checks."""

import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    compute_black_or_white_image_flags,
    count_queries_with_all_gold_black_or_white,
)
from tests.test_tasks.test_task_quality import _image_field_quality

Image = pytest.importorskip("PIL.Image", reason="Image dependencies are not installed")


def test_black_or_white_image_statistics() -> None:
    black_or_white = [
        Image.new(mode, (32, 32), colour)
        for mode, colour in [
            ("RGB", (0, 0, 0)),
            ("RGB", (255, 255, 255)),
            ("L", 0),
            ("L", 255),
            ("RGBA", (0, 0, 0, 255)),
            ("RGBA", (255, 255, 255, 255)),
            ("CMYK", (0, 0, 0, 0)),
            ("CMYK", (0, 0, 0, 255)),
            ("CMYK", (255, 255, 255, 255)),
            ("I;16", 0),
            ("I;16", 65535),
            ("I", 2**31 - 1),
            ("HSV", (0, 0, 0)),
            ("HSV", (0, 0, 255)),
            ("YCbCr", (255, 128, 128)),
        ]
    ]
    other_images = [
        Image.new(mode, (32, 32), colour)
        for mode, colour in [
            ("RGB", (128, 128, 128)),
            ("RGB", (255, 0, 0)),
            ("L", 200),
            ("CMYK", (0, 255, 255, 0)),
            ("I;16", 1000),
            ("F", 1.0),
            ("HSV", (0, 255, 255)),
        ]
    ]
    almost_black = Image.new("RGB", (32, 32))
    almost_black.putpixel((31, 31), (0, 0, 1))
    varying_alpha = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    varying_alpha.putpixel((0, 0), (0, 0, 0, 0))
    palette = Image.new("P", (32, 32))
    palette.putpalette([255, 0, 0, 0, 0, 0, 255, 255, 255] + [0] * 759)
    other_images.extend([almost_black, varying_alpha, palette, palette.convert("PA")])
    for index in (1, 2):
        blank_palette = palette.copy()
        blank_palette.paste(index, (0, 0, 32, 32))
        black_or_white.append(blank_palette)
    mixed_palette = blank_palette.copy()
    mixed_palette.putpixel((0, 0), 1)
    other_images.append(mixed_palette)

    images = black_or_white + other_images
    flags = compute_black_or_white_image_flags(images)
    assert flags == [True] * len(black_or_white) + [False] * len(other_images)
    stats = calculate_image_statistics(images)
    assert stats["black_or_white_images"] == len(black_or_white)
    clean_stats = calculate_image_statistics(other_images)
    assert clean_stats["black_or_white_images"] == 0
    for field in (
        "image_statistics",
        "queries_image_statistics",
        "documents_image_statistics",
    ):
        _, errors = _image_field_quality("test", "test", field, stats)
        assert (f"black_or_white_image:{field}") in dict(errors)
        _, errors = _image_field_quality("test", "test", field, clean_stats)
        assert (f"black_or_white_image:{field}") not in dict(errors)

    relevant_docs = {
        "all_blank": {"0": 1, "1": 2},
        "mixed": {"0": 1, str(len(black_or_white)): 1},
        "zero_not_gold": {"0": 1, str(len(black_or_white)): 0},
        "no_positive": {"0": 0, "1": -1},
        "empty": {},
    }
    blank_ids = {str(i) for i, flag in enumerate(flags) if flag}
    assert count_queries_with_all_gold_black_or_white(relevant_docs, blank_ids) == 2
    assert count_queries_with_all_gold_black_or_white(relevant_docs, {"unused"}) == 0
