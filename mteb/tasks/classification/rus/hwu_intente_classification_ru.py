from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuHWUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuHWUIntentClassification",
        description="",
        dataset={
            "path": "DeepPavlov/hwu_intent_classification_ru",
            "revision": "4335d9ed6ee852568247a7927c197498b7a37ad1",
        },
        reference="https://arxiv.org/abs/1903.05566",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="f1",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""""",
    )
