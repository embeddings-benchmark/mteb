from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RomanianReviewsSentiment(AbsTaskClassification):
    superseded_by = "RomanianReviewsSentiment.v2"
    metadata = TaskMetadata(
        name="RomanianReviewsSentiment",
        description="LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian",
        reference="https://arxiv.org/abs/2101.04197",
        dataset={
            "path": "universityofbucharest/laroseda",
            "revision": "358bcc95aeddd5d07a4524ee416f03d993099b23",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2020-01-01", "2021-01-11"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{tache2101clustering,
  author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
  journal = {ArXiv},
  title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
  year = {2021},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"content": "text", "starRating": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class RomanianReviewsSentimentV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianReviewsSentiment.v2",
        description="""LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://arxiv.org/abs/2101.04197",
        dataset={
            "path": "mteb/romanian_reviews_sentiment",
            "revision": "6b320d55fcf5fc184a9e7cc828debb34f7949432",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2020-01-01", "2021-01-11"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{tache2101clustering,
  author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
  journal = {ArXiv},
  title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
  year = {2021},
}
""",
        adapted_from=["RomanianReviewsSentiment"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
