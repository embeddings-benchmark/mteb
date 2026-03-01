from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakMovieReviewSentimentClassification",
        description="User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)",
        reference="https://arxiv.org/pdf/2304.01922",
        dataset={
            "path": "mteb/SlovakMovieReviewSentimentClassification",
            "revision": "7d408f245f0d3df0eaaca08f89ab6a1936ada6f5",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        date=("2002-05-21", "2020-03-05"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@misc{štefánik2023resources,
  archiveprefix = {arXiv},
  author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
  eprint = {2304.01922},
  primaryclass = {cs.CL},
  title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
  year = {2023},
}
""",
        superseded_by="SlovakMovieReviewSentimentClassification.v2",
    )


class SlovakMovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakMovieReviewSentimentClassification.v2",
        description="User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative) This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/pdf/2304.01922",
        dataset={
            "path": "mteb/slovak_movie_review_sentiment",
            "revision": "29a7405aabcfd4860a51ae6f65a5650d63108f26",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        date=("2002-05-21", "2020-03-05"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@misc{štefánik2023resources,
  archiveprefix = {arXiv},
  author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
  eprint = {2304.01922},
  primaryclass = {cs.CL},
  title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
  year = {2023},
}
""",
        adapted_from=["SlovakMovieReviewSentimentClassification"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
