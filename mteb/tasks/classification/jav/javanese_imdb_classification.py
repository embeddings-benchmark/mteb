from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class JavaneseIMDBClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JavaneseIMDBClassification",
        description="Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.",
        reference="https://github.com/w11wo/nlp-datasets#javanese-imdb",
        dataset={
            "path": "mteb/JavaneseIMDBClassification",
            "revision": "6bf102a9551cc100314a97a470f07b6e43f3f346",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2021-06-24", "2021-06-24"),
        eval_splits=["test"],
        eval_langs=["jav-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wongso2021causal,
  author = {Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
  booktitle = {2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
  organization = {IEEE},
  pages = {1--7},
  title = {Causal and Masked Language Modeling of Javanese Language using Transformer-based Architectures},
  year = {2021},
}
""",
        superseded_by="JavaneseIMDBClassification.v2",
    )


class JavaneseIMDBClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JavaneseIMDBClassification.v2",
        description="Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/w11wo/nlp-datasets#javanese-imdb",
        dataset={
            "path": "mteb/javanese_imdb",
            "revision": "47aadc77049fa4e7b9001c69a255555814d026d9",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2021-06-24", "2021-06-24"),
        eval_splits=["test"],
        eval_langs=["jav-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wongso2021causal,
  author = {Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
  booktitle = {2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
  organization = {IEEE},
  pages = {1--7},
  title = {Causal and Masked Language Modeling of Javanese Language using Transformer-based Architectures},
  year = {2021},
}
""",
        adapted_from=["JavaneseIMDBClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
