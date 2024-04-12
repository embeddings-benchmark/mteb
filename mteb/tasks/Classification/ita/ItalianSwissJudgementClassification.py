from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ItalianSwissJudgementClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ItalianSwissJudgementClassification",
        description="Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)",
        reference="https://aclanthology.org/2021.nllp-1.3/",
        dataset={
            "path": "rcds/swiss_judgment_prediction",
            "revision": "29806f87bba4f23d0707d3b6d9ea5432afefbe2f",
            "language": "it",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=[
            "ita-Latn",
        ],
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
        n_samples={"train": 59709, "validation": 8208, "test": 17357},
        avg_character_length=None,
    )
