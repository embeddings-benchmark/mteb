"""Tests for leaderboard data validation functionality."""

import json
from unittest.mock import patch

import pytest

from mteb.leaderboard.app import _validate_benchmark_json


class TestValidateBenchmarkJson:
    """Test JSON validation for benchmark data."""

    def test_valid_json(self, mock_benchmark_json):
        """Test that valid benchmark JSON passes validation."""
        # Should not raise any exception
        _validate_benchmark_json(mock_benchmark_json)

    def test_invalid_json_syntax(self, mock_invalid_json):
        """Test that invalid JSON syntax raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _validate_benchmark_json(mock_invalid_json)

    def test_non_dict_json(self):
        """Test that non-dictionary JSON raises ValueError."""
        list_json = '["not", "a", "dict"]'
        with pytest.raises(ValueError, match="Expected JSON object, got list"):
            _validate_benchmark_json(list_json)

    def test_missing_results_key(self):
        """Test that missing 'results' key logs warning but doesn't raise."""
        no_results_json = '{"other_key": "value"}'
        with patch("mteb.leaderboard.app.logger.warning") as mock_warning:
            _validate_benchmark_json(no_results_json)
            # Should be called at least once with the missing keys message
            # (may also be called for small JSON warning)
            mock_warning.assert_any_call("Missing expected keys in JSON: ['results']")

    def test_results_not_list(self):
        """Test that non-list 'results' raises ValueError."""
        bad_results_json = '{"results": "not a list"}'
        with pytest.raises(
            ValueError, match="Expected 'results' to be a list, got str"
        ):
            _validate_benchmark_json(bad_results_json)

    def test_small_json_warning(self):
        """Test that unusually small JSON logs warning."""
        small_json = '{"results": []}'  # Less than 1000 chars
        with patch("mteb.leaderboard.app.logger.warning") as mock_warning:
            _validate_benchmark_json(small_json)
            mock_warning.assert_called_once()
            assert "unusually small" in mock_warning.call_args[0][0]

    def test_generic_exception_handling(self):
        """Test that other exceptions are wrapped in ValueError."""
        with patch("json.loads", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(ValueError, match="Invalid benchmark data structure"):
                _validate_benchmark_json('{"test": "data"}')
