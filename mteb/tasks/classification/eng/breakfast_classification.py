from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BreakfastClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BreakfastClassification",
        description="The Breakfast Actions dataset contains 433 videos of 10 breakfast-related activities (e.g. making coffee, preparing cereal, frying eggs) recorded in 18 different kitchens. The task is to classify each video into the correct activity.",
        reference="https://ieeexplore.ieee.org/document/6909500",
        dataset={
            "path": "mteb/Breakfast",
            "revision": "59a874899eb241993794a3454c37829727c3b559",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2014-06-23",
            "2014-06-28",
        ),
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{kuehne2014language,
  author = {Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle = {2014 IEEE Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2014.338},
  pages = {3325-3332},
  title = {The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities},
  year = {2014},
}
""",
    )

    input_column_name = "video"
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True
