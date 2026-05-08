from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoTouche2020Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoTouche2020Retrieval",
        description="NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "zeta-alpha-ai/NanoTouche2020",
            "revision": "0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{potthast_2022_6862281,
  author = {Potthast, Martin and
Gienapp, Lukas and
Wachsmuth, Henning and
Hagen, Matthias and
Fröbe, Maik and
Bondarenko, Alexander and
Ajjour, Yamen and
Stein, Benno},
  doi = {10.5281/zenodo.6862281},
  month = jul,
  publisher = {Zenodo},
  title = {{Touché20-Argument-Retrieval-for-Controversial-
Questions}},
  url = {https://doi.org/10.5281/zenodo.6862281},
  year = {2022},
}
""",
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
        adapted_from=["Touche2020"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "corpus",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "queries",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "qrels",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )

        self.dataset = {}
        for split in corpus_hf:
            corpus_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in corpus_hf[split]]
            )
            queries_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in queries_hf[split]]
            )
            relevant_docs: dict = defaultdict(dict)
            for query_id, corpus_id in zip(
                qrels_hf[split]["query-id"],
                qrels_hf[split]["corpus-id"],
            ):
                relevant_docs[query_id][corpus_id] = 1
            if "default" not in self.dataset:
                self.dataset["default"] = {}
            self.dataset["default"][split] = {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": dict(relevant_docs),
                "top_ranked": None,
            }

        self.data_loaded = True
