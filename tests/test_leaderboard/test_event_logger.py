"""Unit tests for leaderboard EventLogger (Mongo optional, silent degradation)."""

from __future__ import annotations

import atexit
import logging
from unittest.mock import MagicMock, patch

import pytest

import mteb.leaderboard.event_logger.logger as logger_module
from mteb.leaderboard.event_logger.logger import EventLogger

LOGGER_NAME = "mteb.leaderboard.event_logger.logger"


@pytest.fixture(autouse=True)
def _no_event_logger_atexit(monkeypatch: pytest.MonkeyPatch) -> None:
    """EventLogger registers atexit handlers; avoid stacking many across tests."""
    monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: None)


def _shutdown(el: EventLogger) -> None:
    """Release thread pool (atexit is patched out in tests)."""
    el._cleanup()


@pytest.mark.parametrize("mongo_env", [None, "", "   "])
def test_without_usable_mongo_uri_warns_and_disables_storage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mongo_env: str | None,
) -> None:
    """No URI from env/args => WARNING once, storage disabled, no Mongo client."""
    monkeypatch.delenv("MONGO_URI", raising=False)
    if mongo_env is not None:
        monkeypatch.setenv("MONGO_URI", mongo_env)

    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    el = EventLogger(mongo_uri=None)
    try:
        assert el._storage_disabled is True
        assert el._storage is None
        assert any(
            "MONGO_URI not configured" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )
    finally:
        _shutdown(el)


def test_explicit_non_empty_mongo_uri_does_not_disable_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor arg alone can enable the storage code path (lazy connect)."""
    monkeypatch.delenv("MONGO_URI", raising=False)

    el = EventLogger(mongo_uri="mongodb://example.invalid:27017")
    try:
        assert el._storage_disabled is False
        assert el._storage is None  # lazy init until first event
    finally:
        _shutdown(el)


def test_log_when_disabled_does_not_construct_mongodb_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled logger must stay offline: no MongoDBStorage on background write."""
    monkeypatch.delenv("MONGO_URI", raising=False)

    with patch(
        "mteb.leaderboard.event_logger.logger.MongoDBStorage"
    ) as mock_storage_cls:
        el = EventLogger(mongo_uri=None)
        try:
            el.log_page_view(session_id="session-test")
        finally:
            _shutdown(el)
        mock_storage_cls.assert_not_called()


def test_log_with_mock_storage_inserts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: storage is created lazily and receives inserted payloads."""
    monkeypatch.delenv("MONGO_URI", raising=False)

    mock_storage = MagicMock()
    with patch(
        "mteb.leaderboard.event_logger.logger.MongoDBStorage",
        return_value=mock_storage,
    ):
        el = EventLogger(mongo_uri="mongodb://example.invalid:27017")
        try:
            el.log_page_view(session_id="session-abc", benchmark="MTEB-English")
        finally:
            _shutdown(el)
        mock_storage.insert.assert_called_once()
        payload = mock_storage.insert.call_args[0][0]
        assert payload["event_name"] == "page_view"
        assert payload["session_id"] == "session-abc"
        assert payload["benchmark"] == "MTEB-English"


def test_storage_init_failure_warns_and_skips_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If MongoDBStorage fails, log WARNING and do not surface errors to callers."""
    monkeypatch.delenv("MONGO_URI", raising=False)

    with (
        patch(
            "mteb.leaderboard.event_logger.logger.MongoDBStorage",
            side_effect=RuntimeError("connection refused"),
        ),
        patch.object(logger_module.logger, "warning") as mock_warning,
    ):
        el = EventLogger(mongo_uri="mongodb://example.invalid:27017")
        try:
            el.log_page_view(session_id="session-fail")
        finally:
            _shutdown(el)

    init_fail_msgs = [
        c.args[0]
        for c in mock_warning.call_args_list
        if c.args and "EventLogger initialization failed" in str(c.args[0])
    ]
    assert init_fail_msgs, "expected WARNING when MongoDBStorage construction fails"
