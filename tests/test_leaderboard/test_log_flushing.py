"""Tests for leaderboard log flushing functionality."""

import logging
from unittest.mock import Mock, patch

from mteb.leaderboard.app import LogFlusher, _flush_logs


class TestLogFlusher:
    """Test the LogFlusher class and batching logic."""

    def test_init(self):
        flusher = LogFlusher(flush_interval=3)
        assert flusher._flush_interval == 3
        assert flusher._flush_count == 0

    def test_flush_on_interval(self):
        """Test that flushing occurs when flush_count reaches interval."""
        with patch("logging.root.handlers", [Mock()]):
            flusher = LogFlusher(flush_interval=2)

            # First call - shouldn't flush
            flusher.flush_if_needed()
            assert flusher._flush_count == 1

            # Second call - should flush
            flusher.flush_if_needed()
            assert flusher._flush_count == 0  # Reset after flush

            # Check that handlers.flush() was called
            logging.root.handlers[0].flush.assert_called_once()

    def test_flush_on_force(self):
        """Test that flushing occurs immediately when force=True."""
        with patch("logging.root.handlers", [Mock()]):
            flusher = LogFlusher(flush_interval=10)

            # Force flush should work regardless of count
            flusher.flush_if_needed(force=True)
            assert flusher._flush_count == 0  # Reset after flush
            logging.root.handlers[0].flush.assert_called_once()

    def test_flush_on_time_elapsed(self):
        """Test that flushing occurs after 2 seconds regardless of count."""
        with patch("logging.root.handlers", [Mock()]):
            with patch("time.time") as mock_time:
                flusher = LogFlusher(flush_interval=10)
                mock_time.return_value = 0.0  # Initial time
                flusher._last_flush_time = 0.0

                # Simulate 3 seconds passing
                mock_time.return_value = 3.0
                flusher.flush_if_needed()

                assert flusher._flush_count == 0  # Reset after flush
                logging.root.handlers[0].flush.assert_called_once()


class TestFlushLogs:
    """Test the global _flush_logs function."""

    @patch("mteb.leaderboard.app._log_flusher")
    def test_flush_logs_calls_flusher(self, mock_flusher):
        """Test that _flush_logs calls the global flusher."""
        _flush_logs(force=True)
        mock_flusher.flush_if_needed.assert_called_once_with(force=True)

        _flush_logs()
        mock_flusher.flush_if_needed.assert_called_with(force=False)
