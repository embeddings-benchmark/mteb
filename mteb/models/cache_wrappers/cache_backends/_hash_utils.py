from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from PIL import Image


def _hash_item(item: Mapping[str, Any]) -> str:
    item_hash = ""
    if "text" in item:
        item_text: str = item["text"]
        item_hash = hashlib.sha256(item_text.encode()).hexdigest()

    if "image" in item:
        image: Image.Image = item["image"]
        item_hash += hashlib.sha256(image.tobytes()).hexdigest()

    if item_hash == 0:
        raise TypeError(f"Unsupported cache key type: {type(item)}")

    return item_hash
