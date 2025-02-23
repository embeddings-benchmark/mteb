from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mteb.tasks.Retrieval.eng.NFCorpusRetrieval import NFCorpus

if TYPE_CHECKING:
    from mteb.abstasks.text.abs_text_retrieval import AbsTextRetrieval


@pytest.mark.parametrize("task", [NFCorpus()])
def test_abstask_calculate_metadata_metrics(task: AbsTextRetrieval):
    task.calculate_metadata_metrics()
