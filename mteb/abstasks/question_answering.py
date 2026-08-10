"""Answer-mode task: score a system on answering queries over a fixed corpus."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mteb.abstasks.abstask import AbsTask
from mteb.models.answer_systems import AnswerProtocol, ExactMatchJudge
from mteb.types.statistics import SplitDescriptiveStatistics

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset

    from mteb.models.answer_systems import JudgeProtocol
    from mteb.models.models_protocols import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs, ScoresDict

logger = logging.getLogger(__name__)


class QADescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for QuestionAnswering tasks."""

    num_queries: int
    num_documents: int


class AbsTaskQuestionAnswering(AbsTask):
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
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> QADescriptiveStatistics:
        data = self.dataset[hf_subset or "default"][split]
        return QADescriptiveStatistics(
            num_queries=len(data["queries"]), num_documents=len(data["corpus"])
        )

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> ScoresDict:
        if not isinstance(model, AnswerProtocol):
            raise TypeError(
                f"{type(model).__name__} does not implement AnswerProtocol "
                "(index and answer methods)."
            )
        corpus = data_split["corpus"]
        queries = data_split["queries"]
        answers = data_split["answers"]
        model.index(
            corpus,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
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
