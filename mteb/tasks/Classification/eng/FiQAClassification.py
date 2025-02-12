from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FiQAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FiQAClassification",
        dataset={
            "path": "FinanceMTEB/FiQA_ABSA",
            "revision": "afa907ab4c6441afb8ee70bd99802bb707d3d2ab",
        },
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://sites.google.com/view/fiqa/home",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-04-23", "2018-04-27"),
        domains=["Finance"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        descriptive_stats={
            "num_samples": {"test": 352},
            "average_text_length": {"test": 140.9005681818182},
            "unique_labels": {"test": 2},
            "labels": {"test": {"1": {"count": 236}, "0": {"count": 116}}},
        },
    )
