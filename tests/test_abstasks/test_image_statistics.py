"""Tests for black/white-image detection and its evaluation semantics.

A black/white image is pure black or pure white everywhere. `is_black_or_white_image`
decides what counts as one and `ImageStatistics.black_or_white_images` reports how many
a split has, but a black/white *document* only breaks evaluation when it is the whole of
some query's gold set -- which is what `count_queries_with_all_gold_black_or_white` decides.
Other constant colours (e.g. a solid red placeholder) are intentionally left unflagged.
"""

import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    compute_black_or_white_image_flags,
    count_queries_with_all_gold_black_or_white,
    is_black_or_white_image,
)

Image = pytest.importorskip("PIL.Image", reason="Image dependencies are not installed")


def _gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    img.putdata(
        [(i % 256, (i * 7) % 256, (i * 13) % 256) for i in range(width * height)]
    )
    return img


def test_black_image_is_detected() -> None:
    assert is_black_or_white_image(Image.new("RGB", (4, 4), (0, 0, 0)))
    assert is_black_or_white_image(Image.new("L", (4, 4), 0))


def test_white_image_is_detected() -> None:
    assert is_black_or_white_image(Image.new("RGB", (4, 4), (255, 255, 255)))
    assert is_black_or_white_image(Image.new("L", (4, 4), 255))


def test_other_constant_colours_are_not_flagged() -> None:
    """A solid non-black/white colour is common template/placeholder imagery, not a defect."""
    assert not is_black_or_white_image(Image.new("RGB", (4, 4), (128, 128, 128)))
    assert not is_black_or_white_image(Image.new("RGB", (4, 4), (255, 0, 0)))
    assert not is_black_or_white_image(Image.new("L", (4, 4), 200))


def test_non_constant_image_is_not_flagged() -> None:
    assert not is_black_or_white_image(_gradient(4, 4))


def test_a_single_differing_pixel_makes_an_image_not_flagged() -> None:
    almost_black = Image.new("RGB", (8, 8), (0, 0, 0))
    almost_black.putpixel((7, 7), (0, 0, 1))
    assert not is_black_or_white_image(almost_black)


def test_black_or_white_detection_covers_every_band() -> None:
    """Constant black RGB with a varying alpha channel is not flagged."""
    varying_alpha = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
    varying_alpha.putpixel((0, 0), (0, 0, 0, 0))
    assert not is_black_or_white_image(varying_alpha)
    assert is_black_or_white_image(Image.new("RGBA", (2, 2), (0, 0, 0, 255)))
    assert is_black_or_white_image(Image.new("RGBA", (2, 2), (255, 255, 255, 255)))


def test_black_palette_image_is_detected() -> None:
    """Mode P stores indices; the index must be resolved to its actual colour."""
    palette = Image.new("P", (4, 4))
    palette.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    palette.putdata([0] * 16)
    assert is_black_or_white_image(palette)
    palette.putdata([1] * 16)
    assert is_black_or_white_image(palette)


def test_non_black_white_palette_colour_is_not_flagged() -> None:
    """A palette index resolving to red must not be mistaken for black via its raw index 0."""
    palette = Image.new("P", (4, 4))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    palette.putdata([0] * 16)  # index 0 -> red
    assert not is_black_or_white_image(palette)


def test_mixed_palette_image_is_not_flagged() -> None:
    palette = Image.new("P", (4, 4))
    palette.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    palette.putdata([0, 1] * 8)
    assert not is_black_or_white_image(palette)


def test_black_or_white_images_are_counted_in_statistics() -> None:
    images = [
        _gradient(4, 4),
        Image.new("RGB", (4, 4), (0, 0, 0)),
        Image.new("RGB", (4, 4), (255, 255, 255)),
        Image.new("RGB", (4, 4), (255, 0, 0)),  # not flagged: neither black nor white
    ]
    assert calculate_image_statistics(images)["black_or_white_images"] == 2


def test_no_black_or_white_images_is_reported_as_zero() -> None:
    assert (
        calculate_image_statistics([_gradient(4, 4), Image.new("RGB", (4, 4), (0, 255, 0))])[
            "black_or_white_images"
        ]
        == 0
    )


def test_query_whose_only_gold_is_black_or_white_is_counted() -> None:
    """WIT-shaped: one image is the sole gold document for its caption."""
    relevant_docs = {"q1": {"d1": 1}}
    assert count_queries_with_all_gold_black_or_white(relevant_docs, {"d1"}) == 1


def test_query_with_a_normal_gold_alongside_a_black_or_white_one_is_not_counted() -> None:
    """RP2k-shaped: a blank among many positives leaves the query answerable (mixed gold set)."""
    relevant_docs = {"q1": {"d1": 1, "d2": 1, "d3": 1}}
    assert count_queries_with_all_gold_black_or_white(relevant_docs, {"d1"}) == 0


def test_black_or_white_document_no_qrel_references_is_not_counted() -> None:
    """OVEN/WebQA-shaped: an inert distractor no query points at -- not flagged."""
    relevant_docs = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    assert count_queries_with_all_gold_black_or_white(relevant_docs, {"d99"}) == 0


def test_only_positive_judgements_count_as_gold() -> None:
    """A zero-scored qrel is not a positive, so it cannot make a query broken."""
    assert (
        count_queries_with_all_gold_black_or_white({"q1": {"d1": 1, "d2": 0}}, {"d1"})
        == 1
    )
    assert count_queries_with_all_gold_black_or_white({"q1": {"d1": 0}}, {"d1"}) == 0


def test_black_or_white_flags_line_up_with_the_images() -> None:
    images = [
        _gradient(4, 4),
        Image.new("RGB", (4, 4), (0, 0, 0)),
        Image.new("RGB", (4, 4), (7, 7, 7)),  # dark grey: not flagged
        _gradient(2, 2),
    ]
    assert compute_black_or_white_image_flags(images) == [False, True, False, False]
