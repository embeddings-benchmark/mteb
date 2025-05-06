from __future__ import annotations

import base64
import hashlib
import json
import zlib
from typing import Any

# Characters used for base62 encoding (A-Z, a-z, 0-9)
BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def int_to_base62(num: int) -> str:
    """Convert an integer to a base62 string."""
    if num == 0:
        return BASE62_CHARS[0]

    result = ""
    while num:
        num, remainder = divmod(num, 62)
        result = BASE62_CHARS[remainder] + result
    return result


def base62_to_int(string: str) -> int:
    """Convert a base62 string to an integer."""
    result = 0
    for char in string:
        result = result * 62 + BASE62_CHARS.index(char)
    return result


def encode_filter_state(filter_state: dict[str, Any]) -> str:
    """Encode filter state dictionary to a compact base62 string

    Args:
        filter_state: Dictionary with all filter parameters

    Returns:
        A compact base62 encoded string
    """
    json_data = json.dumps(filter_state, separators=(",", ":"))

    compressed = zlib.compress(json_data.encode("utf-8"))
    base64_data = base64.b64encode(compressed).decode("ascii")

    return base64_data.replace("+", "-").replace("/", "_").replace("=", "")


def decode_filter_state(encoded_str: str) -> dict[str, Any]:
    """Decode a base62 encoded string back to filter state dictionary

    Args:
        encoded_str: The compact encoded string

    Returns:
        Dictionary with all filter parameters
    """
    try:
        padding = 4 - (len(encoded_str) % 4)
        if padding < 4:
            encoded_str = encoded_str + ("=" * padding)

        encoded_str = encoded_str.replace("-", "+").replace("_", "/")

        compressed = base64.b64decode(encoded_str)
        json_data = zlib.decompress(compressed).decode("utf-8")

        return json.loads(json_data)
    except Exception as e:
        print(f"Error decoding state: {e}")
        return {}


def generate_short_url(long_url: str) -> str:
    """Generate a short identifier for a URL

    Args:
        long_url: The full benchmark URL with all parameters

    Returns:
        A short identifier (8 characters)
    """
    hash_obj = hashlib.sha256(long_url.encode())
    return hash_obj.hexdigest()[:8]
