from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinancialPhraseBankClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialPhraseBankClassification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://arxiv.org/abs/1307.5336",
        dataset={
            "path": "FinanceMTEB/financial_phrasebank",
            "revision": "14efe6ac2635395e768682e0b91ce794b50c7ff3",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )
