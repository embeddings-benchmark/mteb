from __future__ import annotations

from mteb.abstasks.AbsTaskRegression import AbsTaskRegression
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuSciBenchCitedCountRegression(MultilingualTask, AbsTaskRegression):
    metadata = TaskMetadata(
        name="RuSciBenchCitedCountRegression",
        description="Predicts the number of times a scientific article has been cited by other papers. The prediction is based on the article's title and abstract. The data is sourced from the Russian electronic library of scientific publications (eLibrary.ru) and includes papers with both Russian and English abstracts.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        type="Regression",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "cited_count_ru": ["rus-Cyrl"],
            "cited_count_en": ["eng-Latn"],
        },
        main_score="kendalltau",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        sample_creation="found",
        annotations_creators="derived",
        dialect=None,
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Predict the number of citations for a scientific article based on the title and abstract",
    )

    def dataset_transform(self):
        for subset in self.dataset:
            self.dataset[subset]["train"] = self.dataset[subset][
                "train"
            ].train_test_split(test_size=2048, seed=self.seed)["test"]


class RuSciBenchYearPublRegression(MultilingualTask, AbsTaskRegression):
    metadata = TaskMetadata(
        name="RuSciBenchYearPublRegression",
        description="Predicts the publication year of a scientific article. The prediction is based on the article's title and abstract. The data is sourced from the Russian electronic library of scientific publications (eLibrary.ru) and includes papers with both Russian and English abstracts.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        type="Regression",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "yearpubl_ru": ["rus-Cyrl"],
            "yearpubl_en": ["eng-Latn"],
        },
        main_score="kendalltau",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        sample_creation="found",
        annotations_creators="derived",
        dialect=None,
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Predict paper publitaction year based on the title and abstract",
    )

    def dataset_transform(self):
        for subset in self.dataset:
            self.dataset[subset]["train"] = self.dataset[subset][
                "train"
            ].train_test_split(test_size=2048, seed=self.seed)["test"]
