from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class VieStudentFeedbackClassification(AbsTaskClassification):
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
        license="MIT",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@InProceedings{8573337,
  author={Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
  booktitle={2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
  title={UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis},
  year={2018},
  volume={},
  number={},
  pages={19-24},
  doi={10.1109/KSE.2018.8573337}
}""",
        descriptive_stats={
            "n_samples": {"test": TEST_SAMPLES},
            "avg_character_length": {"test": 14.22},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "sentiment": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
