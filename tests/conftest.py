"""Shared test fixtures and configuration for all tests."""

import gzip
import io
from pathlib import Path

import pytest

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
