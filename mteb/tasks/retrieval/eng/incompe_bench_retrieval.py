from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.types import RelevantDocumentsType


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
) -> tuple[dict[str, Dataset], dict[str, Dataset], dict[str, RelevantDocumentsType]]:
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        queries[split] = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        queries[split] = (
            queries[split]
            .rename_column("query-id", "id")
            .rename_column("query", "text")
            .select_columns(["id", "text"])
        )

        corpus[split] = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        corpus[split] = (
            corpus[split]
            .rename_column("corpus-id", "id")
            .select_columns(["id", "audio"])
        )

        qrels = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        relevant_docs[split] = {}
        for row in qrels:
            relevant_docs[split].setdefault(row["query-id"], {})[row["corpus-id"]] = (
                int(row["score"])
            )

    return corpus, queries, relevant_docs


class IncompeBenchStrictRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IncompeBenchStrictRetrieval",
        description=(
            "IncompeBench-Strict is a fine-grained text-to-music retrieval "
            "benchmark with 500 diverse search queries, 1,574 permissively "
            "licensed music snippets, and graded relevance judgements. The "
            "two variants share the same queries and corpus; unlike Lenient, "
            "Strict drops score-1 tangential matches and retains only scores 2-3."
        ),
        reference="https://arxiv.org/abs/2602.11941",
        dataset={
            "path": "mixedbread-ai/incompebench-strict",
            "revision": "ad7297f765dcff7e398dccefa0f9caf4b5c81715",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-29", "2026-02-12"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{clavie2026incompebench,
  archiveprefix = {arXiv},
  author = {Benjamin Clavi\'e and Atoof Shakir and Jonah Turner and Sean Lee and Aamir Shakir and Makoto P. Kato},
  eprint = {2602.11941},
  primaryclass = {cs.IR},
  title = {IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval},
  url = {https://arxiv.org/abs/2602.11941},
  year = {2026},
}
""",
        prompt={"query": "Retrieve music that matches the user's search query."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.data_loaded = True


class IncompeBenchLenientRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IncompeBenchLenientRetrieval",
        description=(
            "IncompeBench-Lenient is a fine-grained text-to-music retrieval "
            "benchmark with 500 diverse search queries, 1,574 permissively "
            "licensed music snippets, and graded relevance judgements. The "
            "two variants share the same queries and corpus; unlike Strict, "
            "Lenient retains all positive relevance scores 1-3, including "
            "score-1 tangential matches."
        ),
        reference="https://arxiv.org/abs/2602.11941",
        dataset={
            "path": "mixedbread-ai/incompebench-lenient",
            "revision": "2b96667d0163b3d86a4f9ba36da3869ff41e0f88",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-29", "2026-02-12"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{clavie2026incompebench,
  archiveprefix = {arXiv},
  author = {Benjamin Clavi\'e and Atoof Shakir and Jonah Turner and Sean Lee and Aamir Shakir and Makoto P. Kato},
  eprint = {2602.11941},
  primaryclass = {cs.IR},
  title = {IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval},
  url = {https://arxiv.org/abs/2602.11941},
  year = {2026},
}
""",
        prompt={"query": "Retrieve music that matches the user's search query."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.data_loaded = True
