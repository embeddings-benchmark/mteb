"""MultiHop-RAG: multi-hop questions over a shared news corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.data import TaskMeta, from_mteb_retrieval

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.data import AnswerTaskData


def _to_answer_data(
    corpus_rows: Iterable[dict[str, Any]], query_rows: Iterable[dict[str, Any]]
) -> AnswerTaskData:
    """Map MultiHop-RAG rows to AnswerTaskData.

    The corpus ships separately, keyed by article url; gold docs are the evidence
    articles. null_query questions have no evidence (answer: insufficient info).
    """
    documents = {
        r["url"]: {"title": r["title"], "text": r["body"]} for r in corpus_rows
    }
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    relevant: dict[str, dict[str, int]] = {}
    for i, row in enumerate(query_rows):
        qid = str(i)
        questions[qid] = row["query"]
        references[qid] = row["answer"]
        relevant[qid] = {
            e["url"]: 1 for e in row["evidence_list"] if e.get("url") in documents
        }
    return from_mteb_retrieval(documents, questions, relevant, references)


def _load(*, max_questions: int | None = None) -> AnswerTaskData:
    """Load MultiHop-RAG. The corpus is shared; max_questions scopes the questions."""
    from datasets import load_dataset

    revision = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
    corpus = load_dataset(
        "yixuantt/MultiHopRAG", "corpus", split="train", revision=revision
    )
    queries = load_dataset(
        "yixuantt/MultiHopRAG", "MultiHopRAG", split="train", revision=revision
    )
    if max_questions is not None:
        queries = queries.select(range(min(max_questions, len(queries))))
    return _to_answer_data(corpus, queries)


multihop_rag = TaskMeta(
    name="MultiHopRAG",
    description=(
        "Multi-hop questions (inference, comparison, temporal, null) over a shared "
        "corpus of ~600 news articles, short answers, gold evidence articles."
    ),
    reference="https://arxiv.org/abs/2401.15391",
    default_judge="qa_f1",
    loader=_load,
)
