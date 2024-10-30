from __future__ import annotations

import json
from typing import Callable


def json_cache(function: Callable):
    """Caching decorator that can deal with anything json serializable"""
    cached_results = {}

    def wrapper(*args, **kwargs):
        key = json.dumps({"__args": args, **kwargs})
        if key in cached_results:
            return cached_results[key]
        result = function(*args, **kwargs)
        cached_results[key] = result
        return result

    return wrapper
