"""Shared test fixtures and configuration for leaderboard tests."""

import pytest


@pytest.fixture(scope="session")
def leaderboard_test_config():
    """Configuration for leaderboard tests."""
    return {
        "timeout": 300,
        "http_port": 7860,
        "max_file_size_mb": 50,
        "flush_interval": 5,
    }


@pytest.fixture
def mock_benchmark_json():
    """Sample valid benchmark JSON for testing."""
    return '{"results": [{"task": "test_task", "score": 0.85, "model": "test_model"}]}'


@pytest.fixture
def mock_invalid_json():
    """Sample invalid JSON for testing error handling."""
    return '{"results": [invalid json structure}'


@pytest.fixture
def mock_gzipped_content():
    """Generate mock gzipped content for testing."""
    import gzip
    import io

    def _generate_gzipped(text_content: str) -> bytes:
        buffer = io.BytesIO()
        with gzip.open(buffer, "wt", encoding="utf-8") as gz_file:
            gz_file.write(text_content)
        return buffer.getvalue()

    return _generate_gzipped
