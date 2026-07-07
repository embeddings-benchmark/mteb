"""Correctness judges and three-axis aggregation for answer-mode eval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.agentic.interface import AnswerResult, ChatModel


@runtime_checkable
class Judge(Protocol):
    """Scores answer correctness in the range 0 to 1."""

    def score(self, question: str, predicted: str, reference: str) -> float:
        """Grade a predicted answer against a reference answer."""
        ...


def _normalize(text: str) -> str:
    # Lowercase, drop articles and punctuation, collapse whitespace.
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


class ExactMatchJudge:
    """Normalized exact match."""

    def score(self, question: str, predicted: str, reference: str) -> float:  # noqa: PLR6301
        """Return 1.0 on a normalized exact match, else 0.0."""
        return 1.0 if _normalize(predicted) == _normalize(reference) else 0.0


_JUDGE_PROMPT = (
    "Decide if the predicted answer is correct given the reference answer. "
    "End your reply with a single word: YES or NO.\n\n"
    "Question: {question}\nReference: {reference}\nPrediction: {predicted}"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _verdict(text: str) -> float:
    # Drop reasoning blocks, then take the last yes/no word as the verdict.
    words = re.findall(r"[a-z]+", _THINK_RE.sub(" ", text).lower())
    for word in reversed(words):
        if word in {"yes", "no"}:
            return 1.0 if word == "yes" else 0.0
    return 0.0


class LLMJudge:
    """Grades open ended answers with a ChatModel, robust to reasoning output."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    def score(self, question: str, predicted: str, reference: str) -> float:
        """Return 1.0 if the judge deems the prediction correct."""
        prompt = _JUDGE_PROMPT.format(
            question=question, reference=reference, predicted=predicted
        )
        return _verdict(self.model.generate([{"role": "user", "content": prompt}]).text)


@dataclass
class AggregateScores:
    """Three-axis summary over a question set: quality, cost, latency."""

    accuracy: float
    mean_cost_usd: float | None
    total_cost_usd: float | None
    mean_latency_s: float | None
    mean_llm_calls: float
    n: int


def aggregate(
    results: Sequence[AnswerResult], correctness: Sequence[float]
) -> AggregateScores:
    """Reduce per-question results into the three reported axes."""
    n = len(results)
    if n == 0:
        return AggregateScores(0.0, None, None, None, 0.0, 0)
    costs = [r.usage.cost_usd for r in results if r.usage.cost_usd is not None]
    latencies = [r.usage.latency_s for r in results if r.usage.latency_s is not None]
    total_cost = sum(costs) if costs else None
    mean_cost = sum(costs) / len(costs) if costs else None
    return AggregateScores(
        accuracy=sum(correctness) / n,
        mean_cost_usd=mean_cost,
        total_cost_usd=total_cost,
        mean_latency_s=(sum(latencies) / len(latencies)) if latencies else None,
        mean_llm_calls=sum(r.usage.num_llm_calls for r in results) / n,
        n=n,
    )
