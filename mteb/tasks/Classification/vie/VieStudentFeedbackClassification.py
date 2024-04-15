from __future__ import annotations

import random
from collections import Counter

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
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2021-12-26", "2021-12-26"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
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
        n_samples={"test": TEST_SAMPLES},
        avg_character_length={"test": 14.22},
    )

    def dataset_transform(self):
        seed = 42
        random.seed(seed)
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("sentiment", "label")

        for split in ["test"]:
            ds = self.dataset[split]
            # Determine number of classes and samples per class
            class_count = Counter([sample["label"] for sample in ds])
            num_classes = len(class_count)
            total_samples = min(TEST_SAMPLES, len(ds))
            samples_per_class = total_samples // num_classes

            # Try to maintain class balance
            balanced_samples = []
            for label, count in class_count.items():
                indices = [i for i, sample in enumerate(ds) if sample["label"] == label]
                if count <= samples_per_class:
                    balanced_samples.extend(indices)
                else:
                    balanced_samples.extend(random.sample(indices, samples_per_class))

            # Add missing quantity since minority classes might have too few
            if len(balanced_samples) < total_samples:
                extra_samples_needed = total_samples - len(balanced_samples)
                remaining_indices = [
                    i for i in range(len(ds)) if i not in balanced_samples
                ]
                balanced_samples.extend(
                    random.sample(remaining_indices, extra_samples_needed)
                )

            test_data = ds.select(balanced_samples)
            self.dataset["test"] = test_data
            assert (
                len(test_data) == TEST_SAMPLES
            ), f"Exceeded {TEST_SAMPLES} samples for 'test' split."
