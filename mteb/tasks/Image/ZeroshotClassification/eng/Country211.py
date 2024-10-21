from __future__ import annotations

import os

from mteb.abstasks.Image.AbsTaskZeroshotClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class Country211Classification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="Country211ZeroShot",
        description="Classifying images of 211 countries.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_country211",
        dataset={
            "path": "clip-benchmark/wds_country211",
            "revision": "1699f138f0558342a1cbf99f7cf36b4361bb5ebc",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2021-02-26",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@article{radford2021learning,
        title={Learning Transferable Visual Models From Natural Language Supervision},
        author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
        journal={arXiv preprint arXiv:2103.00020},
        year={2021}
        }""",
        descriptive_stats={
            "n_samples": {"test": 21100},
            "avg_character_length": {"test": 0},
        },
    )

    image_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, "templates/Country211_labels.txt")) as f:
            labels = f.readlines()

        return [f"a photo showing the country of {c}." for c in labels]
