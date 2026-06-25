"""Shared narrow exception sets for cache/load paths.

One source of truth so ``frames``, ``routes``, and ``warmup`` can't drift apart.
"""

from __future__ import annotations

import polars as pl

# Excludes ``AttributeError`` / ``TypeError`` so programmer bugs surface.
FRAME_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    pl.exceptions.PolarsError,
)

# Background-preload variant: optional model-meta attrs may be absent.
PRELOAD_ERRORS: tuple[type[BaseException], ...] = (*FRAME_LOAD_ERRORS, AttributeError)
