"""Reference CorpusHandle implementations for the in-process runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import SearchProtocol


class InMemoryCorpus:
    """Read-only corpus backed by a dict of documents, with no search index."""

    def __init__(self, documents: dict[str, dict[str, str]]) -> None:
        # documents maps doc_id to a mapping with at least a text key.
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


def _retrieval_task_metadata(name: str = "AgenticRetrieval") -> TaskMetadata:
    from mteb.abstasks.task_metadata import TaskMetadata

    return TaskMetadata(
        dataset={"path": "mteb/agentic", "revision": "main"},
        name=name,
        description="Answer-mode retrieval over a fixed corpus.",
        type="Retrieval",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )


class RetrievalCorpus(InMemoryCorpus):
    """CorpusHandle backed by any MTEB retriever (a SearchProtocol model).

    Pass mteb.get_model("mteb/baseline-bb25") for BM25, or a
    SearchEncoderWrapper around an encoder for dense. The corpus is indexed
    once at construction; search runs one query through the model per call.
    """

    def __init__(
        self,
        documents: dict[str, dict[str, str]],
        search_model: SearchProtocol,
        *,
        task_metadata: TaskMetadata | None = None,
        encode_kwargs: dict | None = None,
    ) -> None:
        super().__init__(documents)
        from datasets import Dataset

        self._model = search_model
        self._meta = task_metadata or _retrieval_task_metadata()
        self._encode_kwargs = dict(encode_kwargs or {})
        self._corpus = Dataset.from_list(
            [
                {"id": d, "title": doc.get("title", ""), "text": doc.get("text", "")}
                for d, doc in documents.items()
            ]
        )
        self._index()

    def _index(self) -> None:
        self._model.index(
            corpus=self._corpus,
            task_metadata=self._meta,
            hf_split="test",
            hf_subset="default",
            encode_kwargs=self._encode_kwargs,
            num_proc=None,
        )

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top-k documents for one query via the wrapped model."""
        from datasets import Dataset

        queries = Dataset.from_list([{"id": "q", "text": query}])
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

    def _run_search(self, queries: object, top_k: int) -> dict:
        return self._model.search(
            queries=queries,
            task_metadata=self._meta,
            hf_split="test",
            hf_subset="default",
            top_k=top_k,
            encode_kwargs=self._encode_kwargs,
            num_proc=None,
        )


class FileSystemCorpus(InMemoryCorpus):
    """Corpus materialized to files so file-based agents can read it. Exposes root."""

    def __init__(self, documents: dict[str, dict[str, str]], root: str) -> None:
        super().__init__(documents)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for doc_id, doc in documents.items():
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
            (self.root / f"{safe}.txt").write_text(doc.get("text", ""))
