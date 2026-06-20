"""Temporary tests - need linux to run. Integration test: run LanguageBind wrappers against mock MTEB tasks."""

from __future__ import annotations

import pytest

import mteb
from tests.mock_tasks import MockAudioClassification, MockImageClassificationTask

languagebind = pytest.importorskip(
    "languagebind",
    reason="languagebind not installed (decord has no macOS ARM64 wheels)",
)


def test_languagebind_image_on_mock_task():
    from mteb.models.model_implementations.language_bind_models import (
        LanguageBindImageWrapper,
    )

    model = LanguageBindImageWrapper("LanguageBind/LanguageBind_Image")
    results = mteb.evaluate(
        model, MockImageClassificationTask(), cache=None, co2_tracker=False
    )

    assert len(results) == 1
    assert results[0].task_name == "MockImageClassification"
    assert results[0].get_score() is not None


def test_languagebind_audio_on_mock_task():
    from mteb.models.model_implementations.language_bind_models import (
        LanguageBindAudioWrapper,
    )

    model = LanguageBindAudioWrapper("LanguageBind/LanguageBind_Audio_FT")
    results = mteb.evaluate(
        model, MockAudioClassification(), cache=None, co2_tracker=False
    )

    assert len(results) == 1
    assert results[0].task_name == "MockAudioClassification"
    assert results[0].get_score() is not None
