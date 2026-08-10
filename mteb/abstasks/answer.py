"""Answer-mode task: score a system on answering queries over a fixed corpus."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mteb.abstasks.abstask import AbsTask
from mteb.models.answer_systems import ExactMatchJudge

if TYPE_CHECKING:
    from mteb.models.answer_systems import JudgeProtocol
    from mteb.types import ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskAnswer(AbsTask):
    """Task where the model answers each query from a fixed corpus.

    The data split holds a corpus, queries, and reference answers. The model
    implements AnswerProtocol: index(corpus) once, then answer(query_id,
    question) per query. A Judge grades each answer against the reference;
    the main score is accuracy, with total LLM cost reported alongside.

    The judge defaults to normalized exact match; pass judge= at construction
    for LLM grading of open-ended answers.
    """

    abstask_prompt = "Answer the question from the corpus."

    def __init__(self, judge: JudgeProtocol | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.judge = judge or ExactMatchJudge()

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, num_proc: int | None = None
    ) -> dict[str, Any]:
        data = self.dataset[hf_subset or "default"][split]
        return {
            "num_queries": len(data["queries"]),
            "num_documents": len(data["corpus"]),
        }

    def _evaluate_subset(
        self,
        model: Any,
        data_split: dict[str, Any],
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> ScoresDict:
        corpus = data_split["corpus"]
        queries = data_split["queries"]
        answers = data_split["answers"]
        model.index(
            corpus,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=kwargs.get("num_proc"),
        )
        scores: list[float] = []
        costs: list[float] = []
        for row in queries:
            qid, question = row["id"], row["text"]
            try:
                result = model.answer(qid, question)
            except Exception as exc:
                logger.error("query %s failed: %r", qid, exc)
                scores.append(0.0)
                continue
            scores.append(self.judge.score(question, result.text, answers[qid]))
            if result.cost_usd is not None:
                costs.append(result.cost_usd)
        return {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "cost_usd": sum(costs) if costs else 0.0,
            "n_queries": len(scores),
        }
