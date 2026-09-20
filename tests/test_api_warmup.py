"""Tests for the startup cache warmup."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("fastapi")

from mteb.api import warmup


def test_prewarm_list_schemas_uses_the_keys_the_routes_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup must call each cached builder with the arguments its route uses.

    ``functools.lru_cache`` keys on the argument tuple, so warming
    ``_benchmark_schemas_bytes()`` leaves ``_benchmark_schemas_bytes(False)`` —
    the call ``/v1/benchmarks`` makes — a miss.
    """
    calls: dict[str, tuple[Any, ...]] = {}

    def _record(name: str) -> Callable[..., None]:
        def _fn(*args: Any) -> None:
            calls[name] = args

        return _fn

    for name in (
        "_menu_schemas_bytes",
        "_benchmark_schemas_bytes",
        "_filtered_task_schemas_bytes",
        "_filtered_model_schemas_bytes",
    ):
        monkeypatch.setattr(warmup, name, _record(name))

    warmup._prewarm_list_schemas()

    assert calls["_menu_schemas_bytes"] == ()
    assert calls["_benchmark_schemas_bytes"] == (False,)
    assert calls["_filtered_task_schemas_bytes"] == (None,) * 5
    assert calls["_filtered_model_schemas_bytes"] == (None,) * 7 + (False,)
