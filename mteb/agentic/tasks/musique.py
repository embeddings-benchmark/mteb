"""MuSiQue: compositional multi-hop questions with guaranteed distractors."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from mteb.agentic.data import TaskMeta, from_mteb_retrieval

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.data import AnswerTaskData


def _doc_id(title: str, text: str) -> str:
    # Same title can wrap different paragraphs, so key by content.
    return hashlib.md5(f"{title}\n{text}".encode(), usedforsecurity=False).hexdigest()[
        :12
    ]


def _to_answer_data(rows: Iterable[dict[str, Any]]) -> AnswerTaskData:
    """Map MuSiQue rows to AnswerTaskData.

    Each question ships 20 paragraphs (2-4 supporting, the rest distractors);
    the corpus is their deduplicated union and gold docs are the supporting ones.
    """
    documents: dict[str, dict[str, str]] = {}
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    relevant: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = row["id"]
        questions[qid] = row["question"]
        references[qid] = row["answer"]
        gold: dict[str, int] = {}
        for para in row["paragraphs"]:
            doc_id = _doc_id(para["title"], para["paragraph_text"])
            documents.setdefault(
                doc_id, {"title": para["title"], "text": para["paragraph_text"]}
            )
            if para["is_supporting"]:
                gold[doc_id] = 1
        relevant[qid] = gold
    return from_mteb_retrieval(documents, questions, relevant, references)


def _load(
    *, split: str = "validation", max_questions: int | None = None
) -> AnswerTaskData:
    """Load MuSiQue. max_questions scopes both the questions and the corpus."""
    from datasets import load_dataset

    # The authors released MuSiQue-Answerable on GitHub, not the Hub; this is the
    # standard Hub mirror of that release (same fields: paragraphs, is_supporting).
    dataset = load_dataset(
        "dgslibisey/MuSiQue",
        split=split,
        revision="c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
    )
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))
    return _to_answer_data(dataset)


musique = TaskMeta(
    name="MuSiQue",
    description=(
        "Compositional 2-4 hop questions over a 20-paragraph pool per question "
        "(guaranteed distractors), short answers, gold supporting paragraphs."
    ),
    reference="https://arxiv.org/abs/2108.00573",
    default_judge="qa_f1",
    loader=_load,
)
