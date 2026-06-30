"""Runs one AnswerSystem over a question set and scores it on three axes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mteb.agentic.metrics import aggregate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mteb.agentic.interface import AnswerResult, AnswerSystem, CorpusHandle
    from mteb.agentic.metrics import AggregateScores, Judge


@dataclass
class AnswerEvaluationResult:
    """Aggregate scores plus a per-question record."""

    scores: AggregateScores
    per_question: list[dict[str, Any]]


class AnswerEvaluator:
    """Evaluate an answer-mode system over a fixed corpus and question set."""

    def __init__(
        self,
        questions: Mapping[str, str],
        references: Mapping[str, str],
        corpus: CorpusHandle,
        judge: Judge,
    ) -> None:
        # questions and references are keyed by query id.
        self.questions = questions
        self.references = references
        self.corpus = corpus
        self.judge = judge

    def __call__(self, system: AnswerSystem) -> AnswerEvaluationResult:
        """Run the system over every question and return aggregate scores."""
        results: list[AnswerResult] = []
        correctness: list[float] = []
        per_question: list[dict[str, Any]] = []
        for qid, question in self.questions.items():
            start = time.perf_counter()
            result = system.answer(question, self.corpus)
            elapsed = time.perf_counter() - start
            if result.usage.latency_s is None:
                result.usage.latency_s = elapsed
            score = self.judge.score(question, result.answer, self.references[qid])
            results.append(result)
            correctness.append(score)
            per_question.append(
                {
                    "query_id": qid,
                    "answer": result.answer,
                    "correct": score,
                    "cited_doc_ids": result.cited_doc_ids,
                    "latency_s": result.usage.latency_s,
                    "cost_usd": result.usage.cost_usd,
                    "num_llm_calls": result.usage.num_llm_calls,
                }
            )
        return AnswerEvaluationResult(
            scores=aggregate(results, correctness),
            per_question=per_question,
        )
