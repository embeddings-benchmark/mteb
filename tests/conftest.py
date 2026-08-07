"""Shared test fixtures and configuration for all tests."""

import gzip
import io
from pathlib import Path

import polars as pl
import pytest
from datasets import Dataset

from mteb import ResultCache


@pytest.fixture
def mock_mteb_cache_path() -> Path:
    return Path(__file__).parent / "mock_mteb_cache"


@pytest.fixture
def mock_mteb_cache(mock_mteb_cache_path: Path) -> ResultCache:
    return ResultCache(cache_path=mock_mteb_cache_path)


@pytest.fixture
def mock_benchmark_json():
    """Sample valid benchmark JSON for testing."""
    return '{"results": [{"task": "test_task", "score": 0.85, "model": "test_model"}]}'


@pytest.fixture
def mock_gzipped_content():
    """Generate mock gzipped content for testing."""

    def _generate_gzipped(text_content: str) -> bytes:
        buffer = io.BytesIO()
        with gzip.open(buffer, "wt", encoding="utf-8") as gz_file:
            gz_file.write(text_content)
        return buffer.getvalue()

    return _generate_gzipped


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
