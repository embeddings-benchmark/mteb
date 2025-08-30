from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TeluguAndhraJyotiNewsClassification(AbsTaskClassification):
    superseded_by = "TeluguAndhraJyotiNewsClassification.v2"
    metadata = TaskMetadata(
        name="TeluguAndhraJyotiNewsClassification",
        description="A Telugu dataset for 5-class classification of Telugu news articles",
        reference="https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset",
        dataset={
            "path": "mlexplorer008/telugu_news_classification",
            "revision": "3821aa93aa461c9263071e0897234e8d775ad616",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tel-Telu"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"body": "text", "topic": "label"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)


class TeluguAndhraJyotiNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TeluguAndhraJyotiNewsClassification.v2",
        description="""A Telugu dataset for 5-class classification of Telugu news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset",
        dataset={
            "path": "mteb/telugu_andhra_jyoti_news",
            "revision": "032752fb5f3a8c5b0814061c6502e0e8d58ec77c",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tel-Telu"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["TeluguAndhraJyotiNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
