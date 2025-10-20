import hashlib

from PIL import Image

from mteb.types import BatchedInput


def _hash_item(item: BatchedInput) -> str:
    item_hash = ""
    if "text" in item:
        item_hash = hashlib.sha256(item["text"].encode()).hexdigest()

    if "image" in item:
        image: Image.Image = item["image"]
        item_hash += hashlib.sha256(image.tobytes()).hexdigest()

    if item_hash == 0:
        raise TypeError(f"Unsupported cache key type: {type(item)}")

    return item_hash
