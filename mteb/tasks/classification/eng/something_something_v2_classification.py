from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SomethingSomethingV2Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SomethingSomethingV2Classification",
        description="Something-Something V2 contains 220,847 short video clips of humans performing pre-defined basic actions with everyday objects. This subset of 5,444 clips is used for action classification into 174 fine-grained categories.",
        reference="https://developer.qualcomm.com/software/ai-datasets/something-something",
        dataset={
            "path": "mteb/SomethingSomethingV2",
            "revision": "13bbc49a06df3ffe41f3823cf429e2d8d685689f",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-06-01",
            "2017-12-31",
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
@inproceedings{goyal2017something,
  author = {Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yiber, Peter and Gallo, Manuel and Mehri, Ahmed and Bax, Florian and Memisevic, Roland},
  booktitle = {2017 IEEE International Conference on Computer Vision (ICCV)},
  doi = {10.1109/ICCV.2017.622},
  pages = {5843-5851},
  title = {The "Something Something" Video Database for Learning and Evaluating Visual Common Sense},
  year = {2017},
}
""",
    )

    input_column_name = "video"
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True
