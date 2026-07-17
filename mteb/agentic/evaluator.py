"""Runs one AnswerSystem over a question set and scores it on three axes."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.metrics import (
    aggregate,
    calibration_error,
    extract_confidence,
    recall_at,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mteb.agentic.interface import AnswerSystem, CorpusHandle
    from mteb.agentic.metrics import AggregateScores, Judge


@dataclass
class AnswerEvaluationResult:
    """Aggregate scores plus a per-question record for auditing and re-grading."""

    scores: AggregateScores
    per_question: list[dict[str, Any]]


def question_record(
    qid: str,
    result: AnswerResult,
    score: float | None,
    *,
    error: str | None = None,
    gold: list[str] | None = None,
) -> dict[str, Any]:
    """One auditable per-question record; the single source of the record schema."""
    return {
        "query_id": qid,
        "answer": result.answer,
        "correct": score,
        "applicable": result.applicable,
        "error": error,
        "cited_doc_ids": result.cited_doc_ids,
        "recall": recall_at(result.cited_doc_ids, gold),
        "confidence": extract_confidence(result.answer),
        "latency_s": result.usage.latency_s,
        "cost_usd": result.usage.cost_usd,
        "num_llm_calls": result.usage.num_llm_calls,
        "trace": result.trace,
    }


class AnswerEvaluator:
    """Evaluate an answer-mode system over a fixed corpus and question set."""

    def __init__(
        self,
        questions: Mapping[str, str],
        references: Mapping[str, str],
        corpus: CorpusHandle | Callable[[str], CorpusHandle],
        judge: Judge,
        *,
        gold: Mapping[str, list[str]] | None = None,
        max_workers: int = 1,
    ) -> None:
        # questions and references are keyed by query id.
        self.questions = questions
        self.references = references
        self.corpus = corpus
        self.judge = judge
        self.gold = gold  # gold doc ids per qid, for retrieval recall
        self.max_workers = max_workers

    def _run_one(
        self, system: AnswerSystem, qid: str, question: str
    ) -> tuple[AnswerResult, float | None, dict[str, Any]]:
        start = time.perf_counter()
        # One failed question must not kill the run.
        error: str | None = None
        # corpus is a shared handle, or a builder for per-question corpora.
        corpus = self.corpus(qid) if callable(self.corpus) else self.corpus
        try:
            result = system.answer(question, corpus)
            # Not-applicable results score None and are excluded from accuracy.
            score = (
                self.judge.score(question, result.answer, self.references[qid])
                if result.applicable
                else None
            )
        except Exception as exc:
            result = AnswerResult(answer="", usage=Usage())
            score = 0.0
            error = repr(exc)
        elapsed = time.perf_counter() - start
        if result.usage.latency_s is None:
            result.usage.latency_s = elapsed
        gold = self.gold.get(qid) if self.gold else None
        record = question_record(qid, result, score, error=error, gold=gold)
        return result, score, record

    def __call__(self, system: AnswerSystem) -> AnswerEvaluationResult:
        """Run the system over every question and return aggregate scores."""
        items = list(self.questions.items())
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                out = list(
                    pool.map(lambda it: self._run_one(system, it[0], it[1]), items)
                )
        else:
            out = [self._run_one(system, qid, question) for qid, question in items]
        results = [row[0] for row in out]
        correctness = [row[1] for row in out]
        per_question = [row[2] for row in out]
        scores = aggregate(results, correctness)
        recalls = [r["recall"] for r in per_question if r["recall"] is not None]
        scores.mean_recall = sum(recalls) / len(recalls) if recalls else None
        scores.calibration_error = calibration_error(
            [r["confidence"] for r in per_question],
            [r["correct"] for r in per_question],
        )
        return AnswerEvaluationResult(scores=scores, per_question=per_question)
