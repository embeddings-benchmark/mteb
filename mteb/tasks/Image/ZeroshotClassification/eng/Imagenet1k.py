from __future__ import annotations

import os

from mteb.abstasks.Image.AbsTaskZeroshotClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class Imagenet1kClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="Imagenet1kZeroShot",
        description="ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.",
        reference="https://ieeexplore.ieee.org/document/5206848",
        dataset={
            "path": "clip-benchmark/wds_imagenet1k",
            "revision": "b24c7a5a3ef12df09089055d1795e2ce7c7e7397",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2012-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
        journal={2009 IEEE Conference on Computer Vision and Pattern Recognition},
        pages={248--255},
        year={2009},
        organization={Ieee}
        }""",
        descriptive_stats={
            "n_samples": {"test": 37200},
            "avg_character_length": {"test": 0},
        },
    )
    image_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, "templates/Imagenet1k_labels.txt")) as f:
            labels = f.readlines()

        return [f"a photo of {c}." for c in labels]
