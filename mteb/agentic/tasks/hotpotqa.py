"""HotpotQA: two-hop Wikipedia questions in the original distractor setting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.data import TaskMeta, from_per_question

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.data import AnswerTaskData


def _to_answer_data(rows: Iterable[dict[str, Any]]) -> AnswerTaskData:
    """Map HotpotQA distractor rows to AnswerTaskData.

    The original distractor setting: each question has its own 10-paragraph
    corpus (2 gold, 8 distractors), keyed by Wikipedia title, with the
    supporting-fact titles as gold.
    """
    corpora: dict[str, dict[str, dict[str, str]]] = {}
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    relevant: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = row["id"]
        questions[qid] = row["question"]
        references[qid] = row["answer"]
        context = row["context"]
        docs = {
            title: {"title": title, "text": " ".join(sentences)}
            for title, sentences in zip(context["title"], context["sentences"])
        }
        corpora[qid] = docs
        gold = dict.fromkeys(row["supporting_facts"]["title"])
        relevant[qid] = {t: 1 for t in gold if t in docs}
    return from_per_question(corpora, questions, relevant, references)


def _load(
    *, split: str = "validation", max_questions: int | None = None
) -> AnswerTaskData:
    """Load HotpotQA. max_questions scopes both the questions and the corpus."""
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))
    return _to_answer_data(dataset)


hotpotqa = TaskMeta(
    name="HotpotQA",
    description=(
        "Two-hop Wikipedia questions (distractor setting): a corpus pooled from "
        "the question contexts, short answers, gold supporting paragraphs."
    ),
    reference="https://arxiv.org/abs/1809.09600",
    default_judge="qa_f1",
    loader=_load,
)
