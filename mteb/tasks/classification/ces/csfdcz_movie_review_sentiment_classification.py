from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CSFDCZMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDCZMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Czech.",
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
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
@inproceedings{štefánik2023resources,
  address = {Dubrovnik, Croatia},
  author = {{\v{S}}tef{\'a}nik, Michal and Kadl{\v{c}}{\'i}k, Marek and Gramacki, Piotr and Sojka, Petr},
  booktitle = {Proceedings of the 9th Workshop on {S}lavic Natural Language Processing 2023 ({S}lavic{NLP} 2023)},
  doi = {10.18653/v1/2023.bsnlp-1.12},
  editor = {Piskorski, Jakub and Marci{\'n}czuk, Micha{\l} and Nakov, Preslav and Ogrodniczuk, Maciej and Pollak, Senja and P{\v{r}}ib{\'a}{\v{n}}, Pavel and Rybak, Piotr and Steinberger, Josef and Yangarber, Roman},
  month = {may},
  pages = {94--105},
  publisher = {Association for Computational Linguistics},
  title = {Resources and Few-shot Learners for In-context Learning in {S}lavic Languages},
  url = {https://aclanthology.org/2023.bsnlp-1.12/},
  year = {2023},
}
""",
        superseded_by="CSFDCZMovieReviewSentimentClassification.v2",
    )
    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
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
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
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
@inproceedings{štefánik2023resources,
  address = {Dubrovnik, Croatia},
  author = {{\v{S}}tef{\'a}nik, Michal and Kadl{\v{c}}{\'i}k, Marek and Gramacki, Piotr and Sojka, Petr},
  booktitle = {Proceedings of the 9th Workshop on {S}lavic Natural Language Processing 2023 ({S}lavic{NLP} 2023)},
  doi = {10.18653/v1/2023.bsnlp-1.12},
  editor = {Piskorski, Jakub and Marci{\'n}czuk, Micha{\l} and Nakov, Preslav and Ogrodniczuk, Maciej and Pollak, Senja and P{\v{r}}ib{\'a}{\v{n}}, Pavel and Rybak, Piotr and Steinberger, Josef and Yangarber, Roman},
  month = {may},
  pages = {94--105},
  publisher = {Association for Computational Linguistics},
  title = {Resources and Few-shot Learners for In-context Learning in {S}lavic Languages},
  url = {https://aclanthology.org/2023.bsnlp-1.12/},
  year = {2023},
}
""",
        adapted_from=["CSFDCZMovieReviewSentimentClassification"],
    )
    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
