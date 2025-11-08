from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuAtisIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuAtisIntentClassification",
        description="The ATIS Spoken Language Systems Pilot Corpus",
        dataset={
            "path": "DeepPavlov/atis_intent_classification_ru",
            "revision": "79475230b220de6ef378381f94b482a6e98721d3",
        },
        reference="https://huggingface.co/datasets/fathyshalab/atis_intents",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("1990-01-01", "1990-01-01"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""""",
    )
