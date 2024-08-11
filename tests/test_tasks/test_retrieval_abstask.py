from __future__ import annotations

import pytest

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.tasks.Retrieval.eng.NFCorpusRetrieval import NFCorpus


@pytest.mark.parametrize("task", [NFCorpus()])
def test_abstask_calculate_metadata_metrics(task: AbsTaskRetrieval):
    task.calculate_metadata_metrics()
