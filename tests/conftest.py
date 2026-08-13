"""Shared test fixtures and configuration for all tests."""

from pathlib import Path

import pytest

from mteb import ResultCache


@pytest.fixture
def mock_mteb_cache_path() -> Path:
    return Path(__file__).parent / "mock_mteb_cache"


@pytest.fixture
def mock_mteb_cache(mock_mteb_cache_path: Path) -> ResultCache:
    return ResultCache(cache_path=mock_mteb_cache_path)
