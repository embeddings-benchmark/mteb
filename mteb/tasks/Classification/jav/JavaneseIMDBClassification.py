from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class JavaneseIMDBClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JavaneseIMDBClassification",
        description="Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.",
        reference="https://github.com/w11wo/nlp-datasets#javanese-imdb",
        dataset={
            "path": "w11wo/imdb-javanese",
            "revision": "11bef3dfce0ce107eb5e276373dcd28759ce85ee",
        },
        type="Classification",
        category="s2s",
        date=None,
        eval_splits=["test"],
        eval_langs=["jav-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 25_000},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
