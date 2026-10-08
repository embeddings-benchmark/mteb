"""Regression tests for :meth:`AbsTask.filter_languages` script filtering.

See https://github.com/embeddings-benchmark/mteb/issues/5594. Previously a
``script=`` filter was silently ignored: ``contains_script`` was fed a full
language-script code (e.g. ``"rus-Cyrl"``) which never matched the bare script
codes (e.g. ``"Cyrl"``), and with ``languages=None`` ``contains_language``
short-circuited to ``True`` for every subset, so nothing was dropped.
"""

from copy import deepcopy

import pytest

from mteb.mocks.mock_tasks import MockMultilingualSTSTask


def _make_script_filter_task() -> MockMultilingualSTSTask:
    """A multilingual task with both Cyrillic and Latin subsets, incl. a mixed one."""
    task = MockMultilingualSTSTask()
    # deepcopy so we never mutate the shared class-level metadata object.
    task.metadata = deepcopy(task.metadata)
    task.metadata.eval_langs = {
        "ru": ["rus-Cyrl"],
        "sr": ["srp-Cyrl"],
        "en": ["eng-Latn"],
        "de": ["deu-Latn"],
        "de-en": ["deu-Latn", "eng-Latn"],
    }
    return task


def test_filter_script_only_drops_non_cyrillic_subsets():
    task = _make_script_filter_task()
    task.filter_languages(languages=None, script=["Cyrl"])
    assert sorted(task.hf_subsets) == ["ru", "sr"]


def test_filter_language_only_ignores_script():
    task = _make_script_filter_task()
    task.filter_languages(languages=["eng", "rus"])
    assert sorted(task.hf_subsets) == ["de-en", "en", "ru"]


def test_filter_language_and_script_both_must_match():
    task = _make_script_filter_task()
    task.filter_languages(languages=["eng", "rus"], script=["Cyrl"])
    assert sorted(task.hf_subsets) == ["ru"]


def test_filter_language_and_script_mismatch_raises():
    task = _make_script_filter_task()
    # "eng" only occurs in Latin subsets, so no subset matches eng + Cyrl.
    with pytest.raises(ValueError, match="No subsets were found"):
        task.filter_languages(languages=["eng"], script=["Cyrl"])
