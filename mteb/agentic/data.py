"""Adapters that turn retrieval datasets into answer-mode inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class AnswerTaskData:
    """Everything an AnswerEvaluator needs for one split."""

    documents: dict[str, dict[str, str]]  # doc_id -> {title, text}
    questions: dict[str, str]  # qid -> question
    references: dict[str, str]  # qid -> reference answer
    gold_by_qid: dict[str, list[str]]  # qid -> gold doc ids
    gold_by_question: dict[str, list[str]]  # question text -> gold doc ids


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
