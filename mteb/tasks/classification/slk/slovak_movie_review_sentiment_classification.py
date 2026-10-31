from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakMovieReviewSentimentClassification",
        description="User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)",
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
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
        superseded_by="SlovakMovieReviewSentimentClassification.v2",
    )


class SlovakMovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakMovieReviewSentimentClassification.v2",
        description="User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative) This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2023.bsnlp-1.12/",
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
        adapted_from=["SlovakMovieReviewSentimentClassification"],
    )

    def dataset_transform(self, **kwargs: Any) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
