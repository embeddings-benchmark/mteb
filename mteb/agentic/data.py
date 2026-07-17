"""Answer-mode task data model and generic adapters.

AnswerTaskData is what every task loads into and every evaluation consumes.
from_mteb_retrieval and from_mteb_task adapt MTEB-style retrieval data into it.
Concrete tasks live in mteb.agentic.tasks, one module per task, each declaring
a TaskMeta.
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
    gold_by_question: dict[str, list[str]]  # question text -> gold doc ids
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


def from_mteb_retrieval(
    corpus: Mapping[str, Mapping[str, str]],
    queries: Mapping[str, str],
    relevant_docs: Mapping[str, Mapping[str, int]],
    answers: Mapping[str, str],
) -> AnswerTaskData:
    """Build answer-mode data from MTEB-style retrieval fields plus answers."""
    documents = {
        doc_id: {
            "title": doc.get("title", ""),
            "text": doc.get("text", doc.get("body", "")),
        }
        for doc_id, doc in corpus.items()
    }
    questions = dict(queries)
    references = dict(answers)
    gold_by_qid = {
        qid: [doc_id for doc_id, score in rels.items() if score > 0]
        for qid, rels in relevant_docs.items()
    }
    gold_by_question = {
        questions[qid]: ids for qid, ids in gold_by_qid.items() if qid in questions
    }
    return AnswerTaskData(
        documents=documents,
        questions=questions,
        references=references,
        gold_by_qid=gold_by_qid,
        gold_by_question=gold_by_question,
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
    documents_by_qid = {
        qid: {
            doc_id: {
                "title": doc.get("title", ""),
                "text": doc.get("text", doc.get("body", "")),
            }
            for doc_id, doc in docs.items()
        }
        for qid, docs in corpora.items()
    }
    questions = dict(queries)
    references = dict(answers)
    gold_by_qid = {
        qid: [doc_id for doc_id, score in rels.items() if score > 0]
        for qid, rels in relevant_docs.items()
    }
    gold_by_question = {
        questions[qid]: ids for qid, ids in gold_by_qid.items() if qid in questions
    }
    return AnswerTaskData(
        documents={},
        questions=questions,
        references=references,
        gold_by_qid=gold_by_qid,
        gold_by_question=gold_by_question,
        documents_by_qid=documents_by_qid,
    )


def from_mteb_task(
    task: Any,
    *,
    split: str = "test",
    subset: str = "default",
    answer_column: str = "answer",
) -> AnswerTaskData:
    """Build answer-mode data from a loaded MTEB AbsTaskRetrieval.

    Reference answers are read from answer_column on the queries dataset.
    """
    split_data = task.dataset[subset][split]
    corpus_ds = split_data["corpus"]
    queries_ds = split_data["queries"]
    relevant_docs = split_data["relevant_docs"]
    corpus = {
        row["id"]: {
            "title": row.get("title", ""),
            "text": row.get("text", row.get("body", "")),
        }
        for row in corpus_ds
    }
    queries = {row["id"]: row["text"] for row in queries_ds}
    answers = {
        row["id"]: row[answer_column] for row in queries_ds if answer_column in row
    }
    return from_mteb_retrieval(corpus, queries, relevant_docs, answers)
