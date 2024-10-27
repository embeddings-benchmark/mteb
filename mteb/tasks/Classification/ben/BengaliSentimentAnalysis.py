from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BengaliSentimentAnalysis(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliSentimentAnalysis",
        description="dataset contains 3307 Negative reviews and 8500 Positive reviews collected and manually annotated from Youtube Bengali drama.",
        reference="https://data.mendeley.com/datasets/p6zc7krs37/4",
        dataset={
            "path": "Akash190104/bengali_sentiment_analysis",
            "revision": "a4b3685b1854cc26c554dda4c7cb918a36a6fb6c",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ben-Beng"],
        main_score="f1",
        date=("2020-06-24", "2020-11-26"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        bibtex_citation="""@inproceedings{sazzed2020cross,
        title={Cross-lingual sentiment classification in low-resource Bengali language},
        author={Sazzed, Salim},
        booktitle={Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
        pages={50--60},
        year={2020}
        }""",
        descriptive_stats={
            "n_samples": {"train": 11807},
            "avg_character_length": {"train": 69.66},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
