from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class VieStudentFeedbackClassification(AbsTaskClassification):
    superseded_by = "VieStudentFeedbackClassification.v2"
    metadata = TaskMetadata(
        name="VieStudentFeedbackClassification",
        description="A Vietnamese dataset for classification of student feedback",
        reference="https://ieeexplore.ieee.org/document/8573337",
        dataset={
            "path": "uitnlp/vietnamese_students_feedback",
            "revision": "7b56c6cb1c9c8523249f407044c838660df3811a",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2021-12-26", "2021-12-26"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{8573337,
  author = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
  booktitle = {2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
  doi = {10.1109/KSE.2018.8573337},
  number = {},
  pages = {19-24},
  title = {UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis},
  volume = {},
  year = {2018},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "sentiment": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class VieStudentFeedbackClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VieStudentFeedbackClassification.v2",
        description="""A Vietnamese dataset for classification of student feedback
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://ieeexplore.ieee.org/document/8573337",
        dataset={
            "path": "mteb/vie_student_feedback",
            "revision": "9f9451c4aaaa5bf528a90fd430afa128fa748e45",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2021-12-26", "2021-12-26"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{8573337,
  author = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
  booktitle = {2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
  doi = {10.1109/KSE.2018.8573337},
  number = {},
  pages = {19-24},
  title = {UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis},
  volume = {},
  year = {2018},
}
""",
        adapted_from=["VieStudentFeedbackClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
