from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class Caltech101ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Caltech101ZeroShot",
        description="Classifying images of 101 widely varied objects.",
        reference="https://ieeexplore.ieee.org/document/1384978",
        dataset={
            "path": "HuggingFaceM4/Caltech-101",
            "name": "with_background_category",
            "revision": "851374102055782c84f89b1b4e9d128a6568847b",
            "trust_remote_code": True,
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2003-01-01",
            "2004-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@INPROCEEDINGS{1384978,
        author={Li Fei-Fei and Fergus, R. and Perona, P.},
        booktitle={2004 Conference on Computer Vision and Pattern Recognition Workshop},
        title={Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories},
        year={2004},
        volume={},
        number={},
        pages={178-178},
        keywords={Bayesian methods;Testing;Humans;Maximum likelihood estimation;Assembly;Shape;Machine vision;Image recognition;Parameter estimation;Image databases},
        doi={10.1109/CVPR.2004.383}}
        """,
        descriptive_stats={
            "n_samples": {"test": 6084},
            "avg_character_length": {"test": 431.4},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
