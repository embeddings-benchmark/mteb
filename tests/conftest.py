"""Shared test fixtures and configuration for all tests."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest
from datasets import Dataset

from mteb import ResultCache

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Without torch (`make test-core`), only collect the test modules that have tests marked `core`.

    pytest imports a module to read its markers, and most test modules import torch, so collecting
    them on `mteb-core` fails before `-m core` can deselect their tests.
    """
    if _HAS_TORCH or not collection_path.name.startswith("test_"):
        return None
    if (
        collection_path.suffix == ".py"
        and "pytest.mark.core" not in collection_path.read_text("utf-8")
    ):
        return True
    return None


@pytest.fixture
def mock_mteb_cache_path() -> Path:
    return Path(__file__).parent / "mock_mteb_cache"


@pytest.fixture
def mock_mteb_cache(mock_mteb_cache_path: Path) -> ResultCache:
    return ResultCache(cache_path=mock_mteb_cache_path)


def _datasets_supports_dictionary_type() -> bool:
    """True if the installed ``datasets`` can convert Polars Categorical columns.

    ``_to_results_df`` goes through ``Dataset.from_polars`` for categorical
    columns (model_name, task_name, …); older ``datasets`` releases raise
    ``NotImplementedError`` for ``pa.DictionaryType``. Probe once at import.
    """
    try:
        Dataset.from_polars(pl.DataFrame({"x": ["a"]}, schema={"x": pl.Categorical}))
    except NotImplementedError:
        return False
    return True


_skip_if_datasets_too_old = pytest.mark.skipif(
    not _datasets_supports_dictionary_type(),
    reason=(
        "installed `datasets` cannot convert Polars Categorical columns "
        "(pa.DictionaryType -> Features.from_arrow_schema NotImplementedError); "
        "skip the _to_results_df-based parity tests on lowest-pin CI"
    ),
)
