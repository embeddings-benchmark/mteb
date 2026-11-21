from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "ru": ["rus-Cyrl"],
    "en": ["eng-Latn"],
}


class RuSciBenchCiteRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCiteRetrieval",
        dataset={
            "path": "mteb/RuSciBenchCiteRetrieval",
            "revision": "673b6be2681107a1a8f888d12736a514623a2e2a",
        },
        description="This task is focused on Direct Citation Prediction for scientific papers from eLibrary, Russia's largest electronic library of scientific publications. Given a query paper (title and abstract), the goal is to retrieve papers that are directly cited by it from a larger corpus of papers. The dataset for this task consists of 3,000 query papers, 15,000 relevant (cited) papers, and 75,000 irrelevant papers. The task is available for both Russian and English scientific texts.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
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
        prompt={
            "query": "Given a title and abstract of a scientific paper, retrieve the titles and abstracts of other relevant papers",
        },
    )


class RuSciBenchCociteRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCociteRetrieval",
        dataset={
            "path": "mteb/RuSciBenchCociteRetrieval",
            "revision": "00c1d36b5a02fd7f4c7976fb4252069cc1148d30",
        },
        description="This task focuses on Co-citation Prediction for scientific papers from eLibrary, Russia's largest electronic library of scientific publications. Given a query paper (title and abstract), the goal is to retrieve other papers that are co-cited with it. Two papers are considered co-cited if they are both cited by at least 5 of the same other papers. Similar to the Direct Citation task, this task employs a retrieval setup: for a given query paper, all other papers in the corpus that are not co-cited with it are considered negative examples. The task is available for both Russian and English scientific texts.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
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
        prompt={
            "query": "Given a title and abstract of a scientific paper, retrieve the titles and abstracts of other relevant papers",
        },
    )
