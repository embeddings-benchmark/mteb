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
    corpus = {
        "black": Image.new("RGB", (32, 32), (0, 0, 0)),
        "white": Image.new("RGB", (32, 32), (255, 255, 255)),
        "cmyk_white": Image.new("CMYK", (32, 32), (0, 0, 0, 0)),  # no ink
        "grey": Image.new("RGB", (32, 32), (128, 128, 128)),
        "almost_black": almost_black,
    }
    images = list(corpus.values())

    flags = compute_black_or_white_image_flags(images)
    assert flags == [True, True, True, False, False]
    assert calculate_image_statistics(images)["black_or_white_images"] == 3

    # Same pairing as AbsTaskRetrieval does with corpus["id"].
    black_or_white_ids = {
        doc_id for doc_id, flag in zip(corpus, flags, strict=True) if flag
    }
    assert black_or_white_ids == {"black", "white", "cmyk_white"}

    relevant_docs = {
        "every_gold_black_or_white": {"black": 1, "white": 1},
        "one_gold_is_grey": {"black": 1, "grey": 1},
        "grey_judged_irrelevant": {"black": 1, "grey": 0},
    }
    assert (
        count_queries_with_all_gold_black_or_white(relevant_docs, black_or_white_ids)
        == 2
    )
