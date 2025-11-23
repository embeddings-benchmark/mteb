from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CSFDCZMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDCZMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Czech.",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "fewshot-goes-multilingual/cs_csfd-movie-reviews",
            "revision": "dd2ede6faaea338ef6b1e2966f06808656975a23",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2002-06-28", "2020-03-13"),
        eval_splits=["test"],
        eval_langs=["ces-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
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
        superseded_by="CSFDCZMovieReviewSentimentClassification.v2",
    )
    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "rating_int": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )


class CSFDCZMovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDCZMovieReviewSentimentClassification.v2",
        description="The dataset contains 30k user reviews from csfd.cz in Czech. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "mteb/csfdcz_movie_review_sentiment",
            "revision": "bda232f79c949fd881572f7e1b9ad59fd04a6c7c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2002-06-28", "2020-03-13"),
        eval_splits=["test"],
        eval_langs=["ces-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
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
        adapted_from=["CSFDCZMovieReviewSentimentClassification"],
    )
    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
