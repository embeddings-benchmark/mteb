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
