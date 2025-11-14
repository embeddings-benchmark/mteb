from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

N_SAMPLES = 2048


class CSFDSKMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDSKMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Slovak.",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "fewshot-goes-multilingual/sk_csfd-movie-reviews",
            "revision": "23a20c659d868740ef9c54854de631fe19cd5c17",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2002-05-21", "2020-03-05"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
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
        superseded_by="CSFDSKMovieReviewSentimentClassification.v2",
    )

    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "rating_int": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=N_SAMPLES
        )


class CSFDSKMovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDSKMovieReviewSentimentClassification.v2",
        description="The dataset contains 30k user reviews from csfd.cz in Slovak. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "mteb/csfdsk_movie_review_sentiment",
            "revision": "257ee340c1399ab5e038a3aea38877f67940774d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2002-05-21", "2020-03-05"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
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
        adapted_from=["CSFDSKMovieReviewSentimentClassification"],
    )

    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=N_SAMPLES
        )
