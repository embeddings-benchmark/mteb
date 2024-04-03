from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SweRecClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SweRecClassification",
        description="A Swedish dataset for sentiment classification on review",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "mteb/swerec_classification",
            "revision": "b07c6ce548f6a7ac8d546e1bbe197a0086409190",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["sv"],
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
        n_samples={"test": 1024},
        avg_character_length={"test": 318.8},
    )
