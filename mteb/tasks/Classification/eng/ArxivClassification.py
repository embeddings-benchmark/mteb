from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ArxivClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ArxivClassification",
        description="Classification Dataset of Arxiv Papers",
        dataset={
            "path": "ccdv/arxiv-classification",
            "revision": "f9bd92144ed76200d6eb3ce73a8bd4eba9ffdc85",
        },
        reference="https://ieeexplore.ieee.org/document/8675939",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 2500},
        avg_character_length=None,
    )
