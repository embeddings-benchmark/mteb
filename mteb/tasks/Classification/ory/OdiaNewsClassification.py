from __future__ import annotations

import random
from collections import Counter

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata
 

TEST_SAMPLES = 2048


class OdiaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OdiaNewsClassification",
        description="A Odia dataset for 3-class classification of Odia news articles",
        reference="https://github.com/goru001/nlp-for-odia",
        dataset={
            "path": "mlexplorer008/odia_news_classification",
            "revision": "ffb8a34c9637fb20256e8c7be02504d16af4bd6b",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["ory-Orya"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 15200, "test": 157},
        avg_character_length={"train": 4222.22, "test": 4115.14},
    )

    def dataset_transform(self):


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
       
        