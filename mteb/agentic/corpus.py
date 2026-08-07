"""Reference CorpusHandle implementations for the in-process runner."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import SearchProtocol


class InMemoryCorpus:
    """Read-only corpus backed by a dict of documents, with no search index."""

    def __init__(self, documents: dict[str, dict[str, str]]) -> None:
        self._docs = documents

    @property
    def documents(self) -> dict[str, dict[str, str]]:
        """All documents, for raw-access systems (RLM, agents) that read the corpus directly."""
        return self._docs

    def get(self, doc_id: str) -> dict[str, str]:
        """Return one document with its id."""
        return {"id": doc_id, **self._docs[doc_id]}

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Search needs a retriever; InMemoryCorpus has none."""
        raise NotImplementedError("InMemoryCorpus has no index. Use RetrievalCorpus.")


def _retrieval_task_metadata() -> TaskMetadata:
    from mteb.abstasks.task_metadata import TaskMetadata

    return TaskMetadata(
        dataset={"path": "mteb/agentic", "revision": "main"},
        name="AgenticRetrieval",
        description="Answer-mode retrieval over a fixed corpus.",
        type="Retrieval",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )


class RetrieverGuard:
    """Serializes access to one SearchProtocol and tracks which corpus indexed it.

    A SearchProtocol owns a single mutable index. When several RetrievalCorpus
    objects share one retriever (per-question corpora, concurrent workers), the
    guard makes each search run against the index of the corpus that issued it.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.owner: int | None = None


class RetrievalCorpus(InMemoryCorpus):
    """CorpusHandle backed by any MTEB retriever (a SearchProtocol model).

    Pass mteb.get_model("mteb/baseline-bb25") for BM25, or a
    SearchEncoderWrapper around an encoder for dense. The corpus is indexed
    once at construction; search runs one query through the model per call.
    Pass a shared RetrieverGuard when several corpora share one retriever.
    """

    def __init__(
        self,
        documents: dict[str, dict[str, str]],
        search_model: SearchProtocol,
        *,
        guard: RetrieverGuard | None = None,
    ) -> None:
        super().__init__(documents)
        from datasets import Dataset

        self._model = search_model
        self._meta = _retrieval_task_metadata()
        self._guard = guard or RetrieverGuard()
        self._corpus = Dataset.from_list(
            [
                {"id": d, "title": doc.get("title", ""), "text": doc.get("text", "")}
                for d, doc in documents.items()
            ]
        )
        with self._guard.lock:
            self._index()

    def _index(self) -> None:
        self._model.index(
            corpus=self._corpus,
            task_metadata=self._meta,
            hf_split="test",
            hf_subset="default",
            encode_kwargs={},
            num_proc=None,
        )
        self._guard.owner = id(self)

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top-k documents for one query via the wrapped model."""
        from datasets import Dataset

        queries = Dataset.from_list([{"id": "q", "text": query}])
        with self._guard.lock:
            if self._guard.owner != id(self):
                self._index()  # another corpus re-indexed the shared retriever
            try:
                out = self._run_search(queries, top_k)
            except ValueError as exc:
                # Dense/late-interaction wrappers free their index after a search;
                # re-index and retry so per-query search works for every retriever.
                if "indexed" not in str(exc).lower():
                    raise
                self._index()
                out = self._run_search(queries, top_k)
        ranking = out.get("q", {})
        return sorted(ranking.items(), key=lambda kv: -kv[1])[:top_k]

    def _run_search(self, queries: object, top_k: int) -> dict[str, dict[str, float]]:
        return self._model.search(
            queries=queries,
            task_metadata=self._meta,
            hf_split="test",
            hf_subset="default",
            top_k=top_k,
            encode_kwargs={},
            num_proc=None,
        )
