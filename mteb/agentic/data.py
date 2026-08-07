"""Answer-mode task data model and generic adapters.

AnswerTaskData is what every task loads into and every evaluation consumes.
from_mteb_retrieval (shared corpus) and from_per_question (one corpus per
question) adapt MTEB-style retrieval fields into it. Concrete tasks live in
mteb.agentic.tasks, one module per task, each declaring a TaskMeta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass
class AnswerTaskData:
    """Everything an AnswerEvaluator needs for one split."""

    documents: dict[str, dict[str, str]]  # shared corpus: doc_id -> {title, text}
    questions: dict[str, str]  # qid -> question
    references: dict[str, str]  # qid -> reference answer
    gold_by_qid: dict[str, list[str]]  # qid -> gold doc ids
    # Set only when each question has its own corpus (long-context, memory tasks);
    # then documents is empty and corpus_for reads this instead.
    documents_by_qid: dict[str, dict[str, dict[str, str]]] | None = None

    def corpus_for(self, qid: str) -> dict[str, dict[str, str]]:
        """The corpus a question sees: its own if per-question, else the shared one."""
        if self.documents_by_qid is not None:
            return self.documents_by_qid[qid]
        return self.documents


@dataclass
class TaskMeta:
    """Metadata and loader for one answer-mode task (peer of SystemMeta)."""

    name: str
    description: str
    loader: Callable[..., AnswerTaskData]
    reference: str | None = None  # paper or dataset URL
    # Canonical metric for this task: "llm", "qa_f1", "mcq", "oolong", "exact_match".
    # evaluate() uses it when the caller does not pass judge=.
    default_judge: str = "llm"

    def load(self, **kwargs: Any) -> AnswerTaskData:
        """Load the task's data; kwargs pass through to the loader."""
        return self.loader(**kwargs)


def _normalize_docs(
    docs: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    return {
        doc_id: {
            "title": doc.get("title", ""),
            "text": doc.get("text", doc.get("body", "")),
        }
        for doc_id, doc in docs.items()
    }


def _gold_from(relevant_docs: Mapping[str, Mapping[str, int]]) -> dict[str, list[str]]:
    return {
        qid: [doc_id for doc_id, score in rels.items() if score > 0]
        for qid, rels in relevant_docs.items()
    }


def from_mteb_retrieval(
    corpus: Mapping[str, Mapping[str, str]],
    queries: Mapping[str, str],
    relevant_docs: Mapping[str, Mapping[str, int]],
    answers: Mapping[str, str],
) -> AnswerTaskData:
    """Build answer-mode data from MTEB-style retrieval fields plus answers."""
    return AnswerTaskData(
        documents=_normalize_docs(corpus),
        questions=dict(queries),
        references=dict(answers),
        gold_by_qid=_gold_from(relevant_docs),
    )


def from_per_question(
    corpora: Mapping[str, Mapping[str, Mapping[str, str]]],
    queries: Mapping[str, str],
    relevant_docs: Mapping[str, Mapping[str, int]],
    answers: Mapping[str, str],
) -> AnswerTaskData:
    """Build answer-mode data where each question carries its own corpus.

    Mirrors from_mteb_retrieval, but corpora is keyed by qid: qid -> (doc_id ->
    {title, text}). For long-context and memory tasks where retrieval over a
    shared corpus does not apply.
    """
    return AnswerTaskData(
        documents={},
        questions=dict(queries),
        references=dict(answers),
        gold_by_qid=_gold_from(relevant_docs),
        documents_by_qid={qid: _normalize_docs(docs) for qid, docs in corpora.items()},
    )
