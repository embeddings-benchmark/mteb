import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    compute_black_or_white_image_flags,
    count_queries_with_all_gold_black_or_white,
)

Image = pytest.importorskip("PIL.Image", reason="Image dependencies are not installed")


def test_black_or_white_image_statistics() -> None:
    almost_black = Image.new("RGB", (32, 32))
    almost_black.putpixel((31, 31), (0, 0, 1))
    images = [
        Image.new("RGB", (32, 32), (0, 0, 0)),
        Image.new("RGB", (32, 32), (255, 255, 255)),
        Image.new("CMYK", (32, 32), (0, 0, 0, 0)),  # no ink, so white once in RGB
        Image.new("RGB", (32, 32), (128, 128, 128)),
        almost_black,
    ]

    flags = compute_black_or_white_image_flags(images)
    assert flags == [True, True, True, False, False]
    assert calculate_image_statistics(images)["black_or_white_images"] == 3

    blank_ids = {str(i) for i, flag in enumerate(flags) if flag}
    relevant_docs = {
        "all_blank": {"0": 1, "1": 2},
        "mixed": {"0": 1, "3": 1},
        "blank_is_only_positive": {"0": 1, "3": 0},
    }
    assert count_queries_with_all_gold_black_or_white(relevant_docs, blank_ids) == 2
