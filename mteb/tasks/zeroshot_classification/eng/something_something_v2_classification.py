from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification


class SomethingSomethingV2ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="SomethingSomethingV2ZeroShotClassification",
        description="Something-Something V2 contains 220,847 short video clips of humans performing pre-defined basic actions with everyday objects. This subset of 5,444 clips is used for action classification into 174 fine-grained categories.",
        reference="https://developer.qualcomm.com/software/ai-datasets/something-something",
        dataset={
            "path": "mteb/SomethingSomethingV2",
            "revision": "13bbc49a06df3ffe41f3823cf429e2d8d685689f",
        },
        type="VideoZeroshotClassification",
        category="v2t",
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
  author = {Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzy{\'n}ska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and Hoppe, Florian and Thurau, Christian and Bax, Ingo and Memisevic, Roland},
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

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of {name}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]

    def dataset_transform(self, num_proc=None):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            label=self.label_column_name,
            n_samples=2048,
        )
