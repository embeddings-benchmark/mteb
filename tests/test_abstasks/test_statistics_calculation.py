from PIL import Image

from mteb.abstasks._statistics_calculation import (
    calculate_single_input_modality_statistics,
)


def test_single_input_statistics_ignore_missing_images() -> None:
    image = Image.new("RGB", (8, 4), "red")

    statistics = calculate_single_input_modality_statistics(
        {
            "text": ["a", "bb"],
            "image": [None, image],
        }
    )

    assert statistics["text_statistics"]["total_text_length"] == 3
    assert statistics["image_statistics"]["min_image_width"] == 8
    assert statistics["image_statistics"]["min_image_height"] == 4


def test_single_input_statistics_return_none_for_absent_modality() -> None:
    statistics = calculate_single_input_modality_statistics({"image": [None, None]})

    assert statistics["image_statistics"] is None


def test_single_input_statistics_keep_hashes_aligned_after_filtering() -> None:
    image = Image.new("RGB", (8, 4), "red")

    statistics = calculate_single_input_modality_statistics(
        {"image": [image, None, image]},
        hashes={"image": ["first", "missing", "second"]},
    )

    assert statistics["image_statistics"]["unique_images"] == 2
