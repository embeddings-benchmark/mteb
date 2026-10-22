from io import BytesIO

import numpy as np
import pytest

from mteb._content_hashes import hash_image
from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache

Image = pytest.importorskip("PIL.Image")


def _image_pair(variant):
    if variant == "size":
        pixels = bytes(range(24))
        return Image.frombytes("RGB", (2, 4), pixels), Image.frombytes(
            "RGB", (4, 2), pixels
        )
    if variant == "mode":
        return Image.frombytes("RGB", (1, 1), b"\xff\0\0"), Image.frombytes(
            "HSV", (1, 1), b"\xff\0\0"
        )
    first = Image.new("P", (2, 2))
    first.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    second = first.copy()
    if variant == "palette":
        second.putpalette([0, 0, 255] + [0, 0, 0] * 255)
    elif variant == "transparency_bytes":
        second.info["transparency"] = bytes([0, 255])
    else:
        second.info["transparency"] = 0
    return first, second


@pytest.mark.parametrize(
    "variant", ["size", "mode", "palette", "transparency", "transparency_bytes"]
)
def test_image_hash_distinguishes_pixel_interpretation(variant):
    first, second = _image_pair(variant)
    assert first.tobytes() == second.tobytes()
    assert hash_image(first) != hash_image(second)


@pytest.mark.parametrize(
    "variant", ["size", "mode", "palette", "transparency", "transparency_bytes"]
)
@pytest.mark.parametrize("with_text", [False, True])
def test_image_cache_preserves_distinct_vectors(tmp_path, variant, with_text):
    first, second = _image_pair(variant)
    items = [{"image": image} for image in (first, second)]
    if with_text:
        for item in items:
            item["text"] = "same caption"
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cache = NumpyCache(tmp_path, initial_vectors=2)
    cache.add(items[:1], vectors[:1])
    assert cache.get_vector(items[1]) is None
    cache.add(items[1:], vectors[1:])
    cache.save()
    cache.close()
    restored = NumpyCache(tmp_path)
    restored.load()
    for item, expected in zip(items, vectors, strict=True):
        np.testing.assert_array_equal(restored.get_vector(item), expected)
    restored.close()


@pytest.mark.parametrize("variant", ["size", "palette"])
def test_image_hash_matches_copy_and_png_roundtrip(variant):
    image, _ = _image_pair(variant)
    assert hash_image(image) == hash_image(image.copy())
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        assert hash_image(image) == hash_image(decoded)


@pytest.mark.parametrize("variant", ["size", "palette"])
def test_remove_duplicates_keeps_distinct_images(variant):
    from datasets import Dataset, DatasetDict

    from mteb.data_cleaning import remove_duplicates
    from mteb.mocks import MockImageClassificationTask

    first, second = _image_pair(variant)
    task = MockImageClassificationTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {"image": [first, second, first.copy()], "label": [0, 1, 0]}
            )
        }
    )
    task.data_loaded = True
    cleaned = remove_duplicates(task)
    assert cleaned.dataset["test"]["label"] == [0, 1]
