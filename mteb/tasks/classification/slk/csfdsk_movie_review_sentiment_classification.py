from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

N_SAMPLES = 2048


class CSFDSKMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDSKMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Slovak.",
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
        dataset={
            "path": "mteb/CSFDSKMovieReviewSentimentClassification",
            "revision": "029e0a726b46539547e872b90d3073a48ce9bc13",
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
        superseded_by="CSFDSKMovieReviewSentimentClassification.v2",
    )

    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20


class CSFDSKMovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDSKMovieReviewSentimentClassification.v2",
        description="The dataset contains 30k user reviews from csfd.cz in Slovak. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
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
        adapted_from=["CSFDSKMovieReviewSentimentClassification"],
    )

    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self, **kwargs: Any) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=N_SAMPLES
        )
