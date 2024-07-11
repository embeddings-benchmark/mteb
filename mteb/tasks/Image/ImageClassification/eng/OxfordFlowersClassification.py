from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks import AbsTaskImageClassification

class OxfordFlowersClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordFlowersClassification",
        description="Classifying flowers",
        reference="https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train",
        dataset={
            "path": "nelorth/oxford-flowers",
            "revision": "a37b1891609c0376fa81eced756e7863e1bd873b",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""d""",
        n_samples={"test": 400000},
        avg_character_length={"test": 431.4},
    )
