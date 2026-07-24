"""Reference CorpusHandle implementations."""

from __future__ import annotations


class InMemoryCorpus:
    """Read-only corpus backed by a dict of documents."""

    def __init__(self, documents: dict[str, dict[str, str]]) -> None:
        self._docs = documents

    def get(self, doc_id: str) -> dict[str, str]:
        """Return one document with its id."""
        return {"id": doc_id, **self._docs[doc_id]}
