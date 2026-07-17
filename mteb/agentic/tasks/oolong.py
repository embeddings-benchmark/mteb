"""OOLONG: dense-access aggregation over a long context (arXiv 2511.02817)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from mteb.agentic.data import TaskMeta, from_per_question

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.data import AnswerTaskData


def _clean_answer(answer: Any) -> str:
    # Some answers ship as a stringified list, e.g. "['incorrect']".
    text = str(answer)
    if text.startswith("[") and text.endswith("]"):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
    return text


def _to_answer_data(rows: Iterable[dict[str, Any]]) -> AnswerTaskData:
    """Map OOLONG rows to AnswerTaskData.

    Each row is one long context plus an aggregation question. The whole context
    is the corpus for that question (no gold subset, retrieval does not help).
    """
    corpora: dict[str, dict[str, dict[str, str]]] = {}
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    relevant: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = str(row["id"])
        questions[qid] = row["question"]
        references[qid] = _clean_answer(row["answer"])
        corpora[qid] = {"context": {"text": row["context_window_text"]}}
        relevant[qid] = {"context": 1}
    return from_per_question(corpora, questions, relevant, references)


def _load(
    *,
    subset: str = "synth",
    config: str | None = None,
    split: str = "test",
    max_questions: int | None = None,
) -> AnswerTaskData:
    """Load OOLONG. subset is "synth" or "real"; real needs config ("dnd")."""
    from datasets import load_dataset

    repo = (
        "oolongbench/oolong-synth" if subset == "synth" else "oolongbench/oolong-real"
    )
    kwargs = {"name": config} if config else {}
    dataset = load_dataset(repo, split=split, **kwargs)
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))
    return _to_answer_data(dataset)


oolong = TaskMeta(
    name="OOLONG",
    description=(
        "Dense-access aggregation over a long context (counting, most-frequent, "
        "and similar): the whole context must be read, retrieval does not help."
    ),
    reference="https://arxiv.org/abs/2511.02817",
    default_judge="oolong",
    loader=_load,
)
