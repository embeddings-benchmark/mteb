"""Shared test fixtures and configuration for all tests."""

import gzip
import io

import pytest


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
