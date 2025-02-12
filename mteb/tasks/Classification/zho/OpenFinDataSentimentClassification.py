from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OpenFinDataSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OpenFinDataSentimentClassification",
        description="Financial scenario QA dataset incuding sentiment task.",
        dataset={
            "path": "FinanceMTEB/OpenFinDataSentiment",
            "revision": "3494bbd2c652c8e1f021f7e60e9ab79faeec257b",
        },
        reference="https://github.com/open-compass/OpenFinData",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2023-12-29", "2023-12-29"),
        domains=["Finance"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        bibtex_citation="""@misc{OpenFinData2023,
          author = {{Open-Compass}},
          title = {OpenFinData},
          year = {2023},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = url{https://github.com/open-compass/OpenFinData}},
          url = {https://github.com/open-compass/OpenFinData},
          note = {Accessed: YYYY-MM-DD}
        }""",
        descriptive_stats={
            "num_samples": {"test": 23},
            "average_text_length": {"test": 96.30434782608695},
            "unique_labels": {"test": 3},
            "labels": {
                "test": {
                    "0": {"count": 8},
                    "2": {"count": 10},
                    "1": {"count": 5},
                }
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
