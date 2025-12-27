from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class RomanianSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianSentimentClassification",
        description="An Romanian dataset for sentiment classification.",
        reference="https://arxiv.org/abs/2009.08712",
        dataset={
            "path": "mteb/RomanianSentimentClassification",
            "revision": "0f8df3d483924afb9130020e9f36ef09558fc9a1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-09-18", "2020-09-18"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dumitrescu2020birth,
  author = {Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
  journal = {arXiv preprint arXiv:2009.08712},
  title = {The birth of Romanian BERT},
  year = {2020},
}
""",
        superseded_by="RomanianSentimentClassification.v2",
    )


class RomanianSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianSentimentClassification.v2",
        description="An Romanian dataset for sentiment classification. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/2009.08712",
        dataset={
            "path": "mteb/romanian_sentiment",
            "revision": "bf545b83db13cf73ed402749b21a7777e0afdc6a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-09-18", "2020-09-18"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dumitrescu2020birth,
  author = {Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
  journal = {arXiv preprint arXiv:2009.08712},
  title = {The birth of Romanian BERT},
  year = {2020},
}
""",
        adapted_from=["RomanianSentimentClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
