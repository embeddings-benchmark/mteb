"""Answer-mode retrieval benchmark. See README.md."""

from __future__ import annotations

from mteb.agentic.corpus import InMemoryCorpus, RetrievalCorpus
from mteb.agentic.data import (
    AnswerTaskData,
    TaskMeta,
    from_mteb_retrieval,
    from_per_question,
)
from mteb.agentic.evaluate import evaluate
from mteb.agentic.evaluator import AnswerEvaluationResult, AnswerEvaluator
from mteb.agentic.interface import (
    AnswerResult,
    AnswerSystem,
    ChatModel,
    ChatResponse,
    CorpusHandle,
    Message,
    ToolCall,
    Usage,
)
from mteb.agentic.metrics import (
    AggregateScores,
    ExactMatchJudge,
    Judge,
    LLMJudge,
    MultipleChoiceJudge,
    NumericToleranceJudge,
    QAF1Judge,
    aggregate,
    calibration_error,
    extract_confidence,
    recall_at,
    to_scores_dict,
)
from mteb.agentic.models import LiteLLMChatModel, OpenAIChatModel
from mteb.agentic.retrievers import (
    HyDERetriever,
    QueryRewriteRetriever,
    RerankRetriever,
)
from mteb.agentic.systems import (
    ClosedBookSystem,
    FullContextSystem,
    IterativeRAG,
    OracleContextSystem,
    RetrieveThenAnswer,
    RLMSystem,
    SearchAgent,
    SystemMeta,
    WindowedFullContextSystem,
    get_system,
    get_system_meta,
    list_systems,
)
from mteb.agentic.tasks import get_task, get_task_meta, list_tasks

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
    "FullContextSystem",
    "HyDERetriever",
    "InMemoryCorpus",
    "IterativeRAG",
    "Judge",
    "LLMJudge",
    "LiteLLMChatModel",
    "Message",
    "MultipleChoiceJudge",
    "NumericToleranceJudge",
    "OpenAIChatModel",
    "OracleContextSystem",
    "QAF1Judge",
    "QueryRewriteRetriever",
    "RLMSystem",
    "RerankRetriever",
    "RetrievalCorpus",
    "RetrieveThenAnswer",
    "SearchAgent",
    "SystemMeta",
    "TaskMeta",
    "ToolCall",
    "Usage",
    "WindowedFullContextSystem",
    "aggregate",
    "calibration_error",
    "evaluate",
    "extract_confidence",
    "from_mteb_retrieval",
    "from_per_question",
    "get_system",
    "get_system_meta",
    "get_task",
    "get_task_meta",
    "list_systems",
    "list_tasks",
    "recall_at",
    "to_scores_dict",
]
