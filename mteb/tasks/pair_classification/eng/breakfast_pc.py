from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BreakfastPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="BreakfastPairClassification",
        description=(
            "Pair classification on the Breakfast Actions dataset: determining "
            "whether two video clips depict the same breakfast activity category "
            "(10 classes). Same-class / different-class pairs are sampled with a "
            "fixed seed. Built by "
            "scripts/data/breakfast_pair_classification/create_data.py."
        ),
        reference="https://ieeexplore.ieee.org/document/6909500",
        dataset={
            "path": "mteb/Breakfast-PC-V",
            "revision": "5c29a53b9045d7bc862d68ff56db3eeb7aaed069",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-06-23", "2014-06-28"),
        domains=["Activity", "Instructional"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
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
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"
