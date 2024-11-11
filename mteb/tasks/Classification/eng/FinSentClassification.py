from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinSentClassification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://finsent.hkust.edu.hk/",
        dataset={
            "path": "FinanceMTEB/FinSent",
            "revision": "68ee0f0abf596e371ef6a308f685071e3b737bbb",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )
