from collections import defaultdict
from typing import cast

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "ru": ["rus-Cyrl"],
    "en": ["eng-Latn"],
}


def load_ruscibench_data(
    path: str,
    langs: list,
    eval_splits: list,
    revision: str | None = None,
):
    corpus: dict[str, dict[str, dict[str, dict[str, str]] | None]] = {
        lang: dict.fromkeys(eval_splits) for lang in langs
    }
    queries: dict[str, dict[str, dict[str, str] | None]] = {
        lang: dict.fromkeys(eval_splits) for lang in langs
    }
    relevant_docs: dict[str, dict[str, dict[str, dict[str, int]] | None]] = {
        lang: dict.fromkeys(eval_splits) for lang in langs
    }

    for lang in langs:
        lang_corpus = cast(
            datasets.Dataset,
            datasets.load_dataset(path, f"corpus-{lang}", revision=revision),
        )["corpus"]
        lang_queries = cast(
            datasets.Dataset,
            datasets.load_dataset(path, f"queries-{lang}", revision=revision),
        )["queries"]
        lang_qrels = cast(
            datasets.Dataset,
            datasets.load_dataset(path, f"{lang}", revision=revision),
        )["test"]
        corpus[lang] = {
            "test": {
                str(e["_id"]): {"text": e["text"], "title": e["title"]}
                for e in lang_corpus
            }
        }
        queries[lang] = {"test": {str(e["_id"]): e["text"] for e in lang_queries}}
        relevant_docs[lang]["test"] = defaultdict(dict)
        for item in lang_qrels:
            relevant_docs[lang]["test"][str(item["query-id"])].update(
                {str(item["corpus-id"]): item["score"]}
            )
    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class RuSciBenchCiteRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCiteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval",
            "revision": "6cb447d02f41b8b775d5d9df7faf472f44d2f1db",
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

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_ruscibench_data(
            path=self.metadata.dataset["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class RuSciBenchCociteRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCociteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval",
            "revision": "a5da47a245275669d2b6ddf8f96c5338dd2428b4",
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

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_ruscibench_data(
            path=self.metadata.dataset["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
