from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SinhalaNewsSourceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsSourceClassification",
        description="This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).",
        dataset={
            "path": "NLPC-UOM/Sinhala-News-Source-classification",
            "revision": "ac4d14eeb68efbef95e247542d4432ce674faeb1",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2021-02-17", "2022-08-20"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dhananjaya2022,
  author = {Dhananjaya et al.},
  journal = {Year of Publication},
  title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
  year = {2022},
}
""",
        superseded_by="SinhalaNewsSourceClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("comment", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class SinhalaNewsSourceClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsSourceClassification.v2",
        description="This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala). This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/sinhala_news_source",
            "revision": "6902767dbfa6189cbe5f5b5b56ee6300b1702d33",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2021-02-17", "2022-08-20"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dhananjaya2022,
  author = {Dhananjaya et al.},
  journal = {Year of Publication},
  title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
  year = {2022},
}
""",
        adapted_from=["SinhalaNewsSourceClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
