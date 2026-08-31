"""Tests for constant-image detection and its evaluation semantics.

A constant image is a single colour everywhere. `is_constant_image` decides what
counts as one and `ImageStatistics.constant_images` reports how many a split has,
but a constant *document* only breaks evaluation when it is the whole of some
query's gold set -- which is what `count_queries_with_all_gold_constant` decides.
"""

import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    compute_constant_image_flags,
    count_queries_with_all_gold_constant,
    is_constant_image,
)

Image = pytest.importorskip("PIL.Image", reason="Image dependencies are not installed")


def _gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    img.putdata(
        [(i % 256, (i * 7) % 256, (i * 13) % 256) for i in range(width * height)]
    )
    return img


def test_constant_image_is_detected() -> None:
    assert is_constant_image(Image.new("RGB", (4, 4), (0, 0, 0)))
    assert is_constant_image(Image.new("RGB", (4, 4), (255, 255, 255)))
    assert is_constant_image(Image.new("L", (4, 4), 128))
    assert not is_constant_image(_gradient(4, 4))


def test_a_single_differing_pixel_makes_an_image_non_constant() -> None:
    almost = Image.new("RGB", (8, 8), (30, 30, 30))
    almost.putpixel((7, 7), (30, 30, 31))
    assert not is_constant_image(almost)


def test_constant_detection_covers_every_band() -> None:
    """Constant RGB with a varying alpha channel is not a constant image."""
    varying_alpha = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    varying_alpha.putpixel((0, 0), (10, 20, 30, 0))
    assert not is_constant_image(varying_alpha)
    assert is_constant_image(Image.new("RGBA", (2, 2), (10, 20, 30, 255)))


def test_constant_palette_image_is_detected() -> None:
    """Mode P stores indices; one repeated index is still one colour."""
    palette = Image.new("P", (4, 4))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    palette.putdata([0] * 16)
    assert is_constant_image(palette)
    palette.putdata([0, 1] * 8)
    assert not is_constant_image(palette)


def test_constant_images_are_counted_in_statistics() -> None:
    images = [
        _gradient(4, 4),
        Image.new("RGB", (4, 4), (0, 0, 0)),
        Image.new("RGB", (4, 4), (255, 255, 255)),
    ]
    assert calculate_image_statistics(images)["constant_images"] == 2


def test_no_constant_images_is_reported_as_zero() -> None:
    assert calculate_image_statistics([_gradient(4, 4)])["constant_images"] == 0


def test_query_whose_only_gold_is_constant_is_counted() -> None:
    """WIT-shaped: one image is the sole gold document for its caption."""
    relevant_docs = {"q1": {"d1": 1}}
    assert count_queries_with_all_gold_constant(relevant_docs, {"d1"}) == 1


def test_query_with_a_normal_gold_alongside_a_constant_one_is_not_counted() -> None:
    """RP2k-shaped: a blank among many positives leaves the query answerable."""
    relevant_docs = {"q1": {"d1": 1, "d2": 1, "d3": 1}}
    assert count_queries_with_all_gold_constant(relevant_docs, {"d1"}) == 0


def test_constant_document_no_qrel_references_is_not_counted() -> None:
    """OVEN/WebQA-shaped: an inert distractor no query points at."""
    relevant_docs = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    assert count_queries_with_all_gold_constant(relevant_docs, {"d99"}) == 0


def test_only_positive_judgements_count_as_gold() -> None:
    """A zero-scored qrel is not a positive, so it cannot make a query broken."""
    assert count_queries_with_all_gold_constant({"q1": {"d1": 1, "d2": 0}}, {"d1"}) == 1
    assert count_queries_with_all_gold_constant({"q1": {"d1": 0}}, {"d1"}) == 0


def test_constant_flags_line_up_with_the_images() -> None:
    images = [_gradient(4, 4), Image.new("RGB", (4, 4), (7, 7, 7)), _gradient(2, 2)]
    assert compute_constant_image_flags(images) == [False, True, False]
