"""Correctness judges and three-axis aggregation for answer-mode eval."""

from __future__ import annotations

import re
from collections import Counter
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
    """Normalized exact match. Suited to verifiable short answers."""

    def score(self, question: str, predicted: str, reference: str) -> float:  # noqa: PLR6301
        """Return 1.0 on a normalized exact match, else 0.0."""
        return 1.0 if _normalize(predicted) == _normalize(reference) else 0.0


class QAF1Judge:
    """Token-level F1 over normalized answers (SQuAD/HotpotQA/MuSiQue metric)."""

    def score(self, question: str, predicted: str, reference: str) -> float:  # noqa: PLR6301
        """Return the token-overlap F1 of prediction against reference."""
        pred = _normalize(predicted).split()
        gold = _normalize(reference).split()
        if not pred or not gold:
            return 1.0 if pred == gold else 0.0
        overlap = sum((Counter(pred) & Counter(gold)).values())
        if overlap == 0:
            return 0.0
        precision, recall = overlap / len(pred), overlap / len(gold)
        return 2 * precision * recall / (precision + recall)


def _extract_choice(text: str) -> str | None:
    # The letter of a multiple-choice answer (A-D), from common answer phrasings.
    for pattern in (r"answer is[^A-Da-d]*\(?([A-Da-d])\)?", r"\(([A-Da-d])\)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    lead = re.match(r"\s*([A-Da-d])\b", text)
    return lead.group(1).upper() if lead else None


class MultipleChoiceJudge:
    """Accuracy on the selected option letter (LongBench-v2 multiple choice)."""

    def score(self, question: str, predicted: str, reference: str) -> float:  # noqa: PLR6301
        """Return 1.0 if the predicted option letter matches the reference."""
        chosen = _extract_choice(predicted)
        return (
            1.0 if chosen is not None and chosen == _extract_choice(reference) else 0.0
        )


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


class NumericToleranceJudge:
    """OOLONG scoring: numeric answers score 0.75**|pred-ref|, else normalized EM."""

    def score(self, question: str, predicted: str, reference: str) -> float:  # noqa: PLR6301
        """Grade with numeric tolerance when the reference is a number, else EM."""
        target = _first_number(reference)
        if target is None:
            return 1.0 if _normalize(predicted) == _normalize(reference) else 0.0
        guess = _first_number(predicted)
        return 0.75 ** abs(guess - target) if guess is not None else 0.0


# Grading method follows BrowseComp (openai/simple-evals): extract the final
# answer from a possibly verbose response, compare only for a match, and emit
# "correct: yes|no".
_JUDGE_PROMPT = (
    "Grade whether the predicted answer matches the reference answer. Extract the "
    "final answer from the prediction and judge only whether it matches the "
    "reference, ignoring phrasing, order, and extra words. Reply with one line, "
    "exactly 'correct: yes' or 'correct: no'.\n\n"
    "Question: {question}\nReference answer: {reference}\nPrediction: {predicted}"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CORRECT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)


def _verdict(text: str) -> float:
    # Prefer the explicit "correct: yes|no" verdict; fall back to a trailing
    # yes/no so weaker judge models that skip the format still parse.
    cleaned = _THINK_RE.sub(" ", text)
    match = _CORRECT_RE.search(cleaned)
    if match:
        return 1.0 if match.group(1).lower() == "yes" else 0.0
    for word in reversed(re.findall(r"[a-z]+", cleaned.lower())):
        if word in {"yes", "no"}:
            return 1.0 if word == "yes" else 0.0
    return 0.0


class LLMJudge:
    """Grades open ended answers with a ChatModel, BrowseComp-style."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    def score(self, question: str, predicted: str, reference: str) -> float:
        """Return 1.0 if the judge deems the prediction correct."""
        prompt = _JUDGE_PROMPT.format(
            question=question, reference=reference, predicted=predicted
        )
        return _verdict(self.model.generate([{"role": "user", "content": prompt}]).text)


def recall_at(cited: object, gold: object) -> float | None:
    """Fraction of gold documents present in the retrieved/cited set, or None."""
    gold_set = set(gold or [])
    if not gold_set:
        return None
    return len(gold_set & set(cited or [])) / len(gold_set)


_CONFIDENCE_RE = re.compile(r"confidence:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_confidence(text: str) -> float | None:
    """A stated confidence in [0,1] parsed from a BrowseComp-style answer, or None."""
    match = _CONFIDENCE_RE.search(text or "")
    return min(1.0, float(match.group(1)) / 100.0) if match else None


def calibration_error(
    confidences: Sequence[float | None],
    corrects: Sequence[float | None],
    bins: int = 10,
) -> float | None:
    """Expected calibration error over (confidence, correctness) pairs, or None."""
    pairs = [
        (c, y) for c, y in zip(confidences, corrects) if c is not None and y is not None
    ]
    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, float]]] = {}
    for conf, correct in pairs:
        buckets.setdefault(min(bins - 1, int(conf * bins)), []).append((conf, correct))
    total = len(pairs)
    error = 0.0
    for members in buckets.values():
        avg_conf = sum(c for c, _ in members) / len(members)
        avg_acc = sum(y for _, y in members) / len(members)
        error += (len(members) / total) * abs(avg_acc - avg_conf)
    return error


@dataclass
class AggregateScores:
    """Three-axis summary over a question set: quality, cost, latency.

    All axes are computed over the questions the system could attempt;
    coverage is the fraction it could.
    """

    accuracy: float
    mean_cost_usd: float | None
    total_cost_usd: float | None
    mean_latency_s: float | None
    mean_llm_calls: float
    n: int
    coverage: float = 1.0
    n_applicable: int = 0
    mean_recall: float | None = None  # gold-doc coverage (BrowseComp-Plus Recall)
    calibration_error: float | None = None  # ECE over stated confidences


def aggregate(
    results: Sequence[AnswerResult], correctness: Sequence[float | None]
) -> AggregateScores:
    """Reduce per-question results into the reported axes, excluding N/A questions."""
    n = len(results)
    if n == 0:
        return AggregateScores(0.0, None, None, None, 0.0, 0, coverage=0.0)
    applicable = [r for r in results if r.applicable]
    n_app = len(applicable)
    scores = [c for c in correctness if c is not None]
    costs = [r.usage.cost_usd for r in applicable if r.usage.cost_usd is not None]
    latencies = [r.usage.latency_s for r in applicable if r.usage.latency_s is not None]
    total_cost = sum(costs) if costs else None
    return AggregateScores(
        accuracy=(sum(scores) / n_app) if n_app else 0.0,
        mean_cost_usd=(total_cost / len(costs)) if costs else None,
        total_cost_usd=total_cost,
        mean_latency_s=(sum(latencies) / len(latencies)) if latencies else None,
        mean_llm_calls=(sum(r.usage.num_llm_calls for r in applicable) / n_app)
        if n_app
        else 0.0,
        n=n,
        coverage=n_app / n,
        n_applicable=n_app,
    )


def to_scores_dict(scores: AggregateScores) -> dict[str, float]:
    """Convert aggregate scores into an MTEB-style scores dict.

    accuracy is the main_score; cost, latency, and coverage ride along as
    extra keys in the result JSON.
    """
    out: dict[str, float] = {
        "accuracy": scores.accuracy,
        "mean_llm_calls": scores.mean_llm_calls,
        "n": float(scores.n),
        "coverage": scores.coverage,
        "n_applicable": float(scores.n_applicable),
    }
    if scores.mean_cost_usd is not None:
        out["mean_cost_usd"] = scores.mean_cost_usd
        if scores.total_cost_usd is not None:
            out["total_cost_usd"] = scores.total_cost_usd
    if scores.mean_latency_s is not None:
        out["mean_latency_s"] = scores.mean_latency_s
    if scores.mean_recall is not None:
        out["mean_recall"] = scores.mean_recall
    if scores.calibration_error is not None:
        out["calibration_error"] = scores.calibration_error
    return out
