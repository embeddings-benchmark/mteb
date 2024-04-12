from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NoRecClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NoRecClassification",
        description="A Norwegian dataset for sentiment classification on review",
        reference="https://aclanthology.org/L18-1661/",
        dataset={
            # using the mini version to keep results ~comparable to the ScandEval benchmark
            "path": "mteb/norec_classification",
            "revision": "5b740b7c42c73d586420812a35745fc37118862f",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 2050},
        avg_character_length={"test": 82},
    )
