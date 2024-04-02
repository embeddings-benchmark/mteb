from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class AmazonPolarityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityClassification",
        description="Amazon Polarity Classification Dataset.",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "mteb/amazon_polarity",
            "revision": "e2d317d38cd51312af73b3d32a06d1a08b442046",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
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
        n_samples={"test": 400000},
        avg_character_length={"test": 431.4},
    )
