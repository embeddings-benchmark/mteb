"""Answer-mode retrieval benchmark (core). See README.md."""

from __future__ import annotations

from mteb.agentic.corpus import InMemoryCorpus
from mteb.agentic.data import AnswerTaskData, from_mteb_retrieval
from mteb.agentic.evaluator import AnswerEvaluationResult, AnswerEvaluator
from mteb.agentic.interface import (
    AnswerResult,
    AnswerSystem,
    ChatModel,
    ChatResponse,
    CorpusHandle,
    Message,
    Usage,
)
from mteb.agentic.metrics import (
    AggregateScores,
    ExactMatchJudge,
    Judge,
    LLMJudge,
    aggregate,
)
from mteb.agentic.systems import ClosedBookSystem, OracleContextSystem

__all__ = [
    "AggregateScores",
    "AnswerEvaluationResult",
    "AnswerEvaluator",
    "AnswerResult",
    "AnswerSystem",
    "AnswerTaskData",
    "ChatModel",
    "ChatResponse",
    "ClosedBookSystem",
    "CorpusHandle",
    "ExactMatchJudge",
    "InMemoryCorpus",
    "Judge",
    "LLMJudge",
    "Message",
    "OracleContextSystem",
    "Usage",
    "aggregate",
    "from_mteb_retrieval",
]
