"""LongBench-v2: long-context multiple-choice QA over one document per question."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.data import TaskMeta, from_per_question

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.data import AnswerTaskData

_CHOICES = ("A", "B", "C", "D")


def _to_answer_data(rows: Iterable[dict[str, Any]]) -> AnswerTaskData:
    """Map LongBench-v2 rows to AnswerTaskData.

    Each row is one long context and a 4-way multiple-choice question. The
    context is the whole corpus for that question; the reference is the correct
    option (letter and text) so the judge can match either form.
    """
    corpora: dict[str, dict[str, dict[str, str]]] = {}
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    relevant: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = str(row["_id"])
        options = "\n".join(f"({c}) {row[f'choice_{c}']}" for c in _CHOICES)
        questions[qid] = f"{row['question']}\n\n{options}"
        letter = row["answer"].strip()
        references[qid] = f"({letter}) {row[f'choice_{letter}']}"
        corpora[qid] = {"context": {"text": row["context"]}}
        relevant[qid] = {"context": 1}
    return from_per_question(corpora, questions, relevant, references)


def _load(*, split: str = "train", max_questions: int | None = None) -> AnswerTaskData:
    """Load LongBench-v2 (all 503 questions ship in the single train split)."""
    from datasets import load_dataset

    dataset = load_dataset("THUDM/LongBench-v2", split=split)
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))
    return _to_answer_data(dataset)


longbench_v2 = TaskMeta(
    name="LongBenchV2",
    description=(
        "Long-context multiple-choice QA (single/multi-document, code, dialogue) "
        "over one context per question, up to ~2M words; scored by the chosen option."
    ),
    reference="https://arxiv.org/abs/2412.15204",
    default_judge="mcq",
    loader=_load,
)
