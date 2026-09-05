from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from PIL import Image


def _hash_item(item: Mapping[str, Any]) -> str:
    """Build the cache key for one dataset row.

    A row of an interleaved dataset carries no value for the modalities it does not
    use; those contribute nothing to the key, exactly as an absent column does. The
    key of a row with a value is unchanged, so existing on-disk caches stay valid.
    """
    item_hash = ""
    item_text: str | None = item.get("text")
    if item_text is not None:
        item_hash = hashlib.sha256(item_text.encode()).hexdigest()

    image: Image.Image | None = item.get("image")
    if image is not None:
        item_hash += hashlib.sha256(image.tobytes()).hexdigest()

    if len(item_hash) == 0:
        raise TypeError(f"Unsupported cache key type: {type(item)}")

    return item_hash
