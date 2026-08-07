"""Front door: evaluate one or several systems on a task.

evaluate() resolves tasks, models, and systems (by registry name or as objects),
builds each required corpus representation once, and runs every system through
AnswerEvaluator. Every paradigm, external agents included, is scored by the same
Judge.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from mteb.agentic.corpus import InMemoryCorpus, RetrievalCorpus, RetrieverGuard
from mteb.agentic.data import AnswerTaskData
from mteb.agentic.evaluator import (
    AnswerEvaluationResult,
    AnswerEvaluator,
    _finalize,
    _question_record,
)
from mteb.agentic.harbor import (
    read_harbor_answers,
    read_harbor_metrics,
    run_harbor,
    to_harbor_dataset,
)
from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.metrics import (
    ExactMatchJudge,
    LLMJudge,
    MultipleChoiceJudge,
    NumericToleranceJudge,
    QAF1Judge,
)
from mteb.agentic.models import OpenAIChatModel
from mteb.agentic.systems import get_system_meta
from mteb.agentic.tasks import get_task_meta

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mteb.agentic.data import TaskMeta
    from mteb.agentic.interface import AnswerSystem, ChatModel, CorpusHandle
    from mteb.agentic.metrics import Judge
    from mteb.agentic.systems import SystemMeta
    from mteb.models.models_protocols import EncoderProtocol, SearchProtocol

# Batch-level kwargs consumed by the Harbor path rather than a system __init__.
_HARBOR_KWARGS = frozenset(
    {"agent_timeout_s", "agent_retriever", "n_concurrent", "agent_env"}
)


@overload
def evaluate(
    system: str | AnswerSystem,
    task: str | AnswerTaskData,
    *,
    systems: None = None,
    model: ChatModel | str | None = None,
    retriever: SearchProtocol | str | None = None,
    judge: Judge | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    work_dir: str | Path | None = None,
    **system_kwargs: Any,
) -> AnswerEvaluationResult: ...


@overload
def evaluate(
    system: None = None,
    task: str | AnswerTaskData | None = None,
    *,
    systems: Sequence[str | AnswerSystem],
    model: ChatModel | str | None = None,
    retriever: SearchProtocol | str | None = None,
    judge: Judge | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    work_dir: str | Path | None = None,
    **system_kwargs: Any,
) -> dict[str, AnswerEvaluationResult]: ...


def evaluate(
    system: str | AnswerSystem | None = None,
    task: str | AnswerTaskData | None = None,
    *,
    systems: Sequence[str | AnswerSystem] | None = None,
    model: ChatModel | str | None = None,
    retriever: SearchProtocol | str | None = None,
    judge: Judge | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    work_dir: str | Path | None = None,
    **system_kwargs: Any,
) -> AnswerEvaluationResult | dict[str, AnswerEvaluationResult]:
    """Evaluate one or several answer-mode systems on one task.

    Pass exactly one of ``system=`` or ``systems=``. The batch form reuses the
    loaded task and compatible corpus representations, including a retrieval
    index, across systems.

    Args:
        system: One system, by registry name or as an AnswerSystem object.
        task: Task registry name or an AnswerTaskData.
        systems: Several systems, keyed by name in the returned dict.
        model: A ChatModel, or a model name built into an OpenAIChatModel using
            OPENAI_BASE_URL / OPENAI_API_KEY from the environment.
        retriever: First-stage corpus for retrieval systems: "bm25", an MTEB
            model name, or a SearchProtocol.
        judge: Correctness judge; defaults to the task's canonical judge.
        limit: Evaluate only the first N questions.
        max_workers: Questions evaluated concurrently (in-process systems).
        work_dir: Where Harbor datasets and job outputs are written.
        **system_kwargs: Extra system arguments (top_k, agent_env, ...), passed
            to each system that accepts them; unknown names raise.

    Returns:
        An AnswerEvaluationResult, or a dict of them keyed by system name when
        systems= is used.
    """
    if task is None:
        raise TypeError("Pass task= a task name or AnswerTaskData.")
    if (system is None) == (systems is None):
        raise ValueError("Pass exactly one of system= or systems=.")
    requested = [system] if system is not None else list(systems or [])
    if not requested:
        raise ValueError("systems= must contain at least one system.")

    if isinstance(task, AnswerTaskData):
        task_meta = None
        data = task
    else:
        task_meta = get_task_meta(task)
        data = task_meta.load()
    questions, references = _select_questions(data, limit)
    judge = judge or _default_judge(task_meta, model)
    corpora = _CorpusPool(data, retriever)
    resolved_model: ChatModel | None = None
    consumed_kwargs: set[str] = set()
    output: dict[str, AnswerEvaluationResult] = {}

    for requested_system in requested:
        key = _system_name(requested_system)
        if key in output:
            raise ValueError(f"Duplicate system name in one evaluation: {key!r}.")
        if isinstance(requested_system, str):
            meta = get_system_meta(requested_system)
            if meta.kind == "harbor":
                consumed_kwargs |= _HARBOR_KWARGS & set(system_kwargs)
                output[key] = _run_harbor_batch(
                    meta,
                    data,
                    questions,
                    references,
                    model=model,
                    judge=judge,
                    work_dir=Path(work_dir) / key
                    if systems is not None and work_dir is not None
                    else work_dir,
                    **{k: v for k, v in system_kwargs.items() if k in _HARBOR_KWARGS},
                )
                continue
            if resolved_model is None:
                resolved_model = _resolve_model(model)
            corpus_kind = meta.corpus_kind
            answer_system = _build_system(
                meta, resolved_model, data, system_kwargs, consumed_kwargs
            )
        else:
            # A prebuilt system chooses its own corpus access via retriever=.
            corpus_kind = "retrieval" if retriever is not None else "memory"
            answer_system = requested_system
            consumed_kwargs |= set(system_kwargs)

        output[key] = AnswerEvaluator(
            questions,
            references,
            corpora.get(corpus_kind),
            judge,
            gold=data.gold_by_qid,
            max_workers=max_workers,
        )(answer_system)

    _reject_unknown_kwargs(system_kwargs, consumed_kwargs)
    return output if systems is not None else next(iter(output.values()))


def _reject_unknown_kwargs(
    system_kwargs: Mapping[str, Any], consumed: set[str]
) -> None:
    unknown = set(system_kwargs) - consumed
    if unknown:
        raise TypeError(f"Unknown system kwargs for this evaluation: {sorted(unknown)}")


class _CorpusPool:
    """Lazily build and reuse corpus views during one batch evaluation."""

    def __init__(
        self, data: AnswerTaskData, retriever: SearchProtocol | str | None
    ) -> None:
        self.data = data
        self.retriever = retriever
        # One guard per pool: every corpus sharing this retriever serializes on it.
        self.guard = RetrieverGuard()
        self._shared: dict[str, CorpusHandle | Callable[[str], CorpusHandle]] = {}

    def get(self, corpus_kind: str) -> CorpusHandle | Callable[[str], CorpusHandle]:
        if corpus_kind in self._shared:
            return self._shared[corpus_kind]

        corpus: CorpusHandle | Callable[[str], CorpusHandle]
        if self.data.documents_by_qid is None:
            corpus = self._build(corpus_kind, self.data.documents)
        else:
            per_question: dict[str, CorpusHandle] = {}

            def resolve(qid: str) -> CorpusHandle:
                # A SearchProtocol owns one mutable index. Do not retain
                # per-question retrieval handles that may share and overwrite it.
                if corpus_kind != "retrieval" and qid in per_question:
                    return per_question[qid]
                built = self._build(corpus_kind, self.data.corpus_for(qid))
                if corpus_kind != "retrieval":
                    per_question[qid] = built
                return built

            corpus = resolve

        self._shared[corpus_kind] = corpus
        return corpus

    def _build(
        self, corpus_kind: str, documents: Mapping[str, dict[str, str]]
    ) -> CorpusHandle:
        if corpus_kind == "retrieval":
            if self.retriever is None:
                raise ValueError(
                    "This system uses a first-stage retriever; pass retriever= "
                    "('bm25', an mteb model name, or a SearchProtocol)."
                )
            return RetrievalCorpus(
                dict(documents), _resolve_retriever(self.retriever), guard=self.guard
            )
        return InMemoryCorpus(dict(documents))


def _system_name(system: str | AnswerSystem) -> str:
    if isinstance(system, str):
        return system
    name = getattr(system, "name", None)
    if not name:
        raise ValueError("Prebuilt systems in systems= must have a non-empty name.")
    return str(name)


def _select_questions(
    data: AnswerTaskData, limit: int | None
) -> tuple[dict[str, str], dict[str, str]]:
    items = list(data.questions.items())
    if limit is not None:
        items = items[:limit]
    questions = dict(items)
    missing = [qid for qid in questions if qid not in data.references]
    if missing:
        raise ValueError(f"Questions with no reference answer: {missing[:5]}")
    references = {qid: data.references[qid] for qid in questions}
    return questions, references


def _resolve_model(model: ChatModel | str | None) -> ChatModel:
    if model is None:
        raise ValueError(
            "Pass model= a ChatModel or a model name (e.g. OpenAIChatModel(...) "
            "or 'Qwen/Qwen3-...' with OPENAI_BASE_URL/OPENAI_API_KEY set)."
        )
    if isinstance(model, str):
        return OpenAIChatModel(model)  # endpoint and key come from the environment
    return model


# The judge behind each TaskMeta.default_judge name (the "llm" metric is
# handled separately because it needs a ChatModel).
_METRIC_JUDGES: dict[str, Callable[[], Judge]] = {
    "qa_f1": QAF1Judge,
    "mcq": MultipleChoiceJudge,
    "oolong": NumericToleranceJudge,
    "exact_match": ExactMatchJudge,
}


def _default_judge(task_meta: TaskMeta | None, model: ChatModel | str | None) -> Judge:
    """The task's canonical judge when the caller passes none.

    Ad-hoc AnswerTaskData defaults to exact match. A task graded by an LLM
    judge requires a usable ChatModel rather than silently downgrading.
    """
    if task_meta is None:
        return ExactMatchJudge()
    metric = task_meta.default_judge
    judge_cls = _METRIC_JUDGES.get(metric)
    if judge_cls is not None:
        return judge_cls()
    if isinstance(model, str) or model is None:
        raise ValueError(
            f"Task {task_meta.name!r} grades with an LLM judge; pass judge= "
            "(e.g. LLMJudge(chat_model)) or model= a ChatModel."
        )
    return LLMJudge(model)


def _resolve_retriever(retriever: SearchProtocol | str) -> SearchProtocol:
    model: object = retriever
    if isinstance(retriever, str):
        import mteb

        name = "mteb/baseline-bb25" if retriever == "bm25" else retriever
        model = mteb.get_model(name)
    if hasattr(model, "index") and hasattr(model, "search"):
        return cast("SearchProtocol", model)  # already a SearchProtocol (e.g. BM25)
    from mteb.models.search_wrappers import SearchEncoderWrapper

    return SearchEncoderWrapper(
        cast("EncoderProtocol", model)
    )  # wrap a plain encoder for dense retrieval


def _build_system(
    meta: SystemMeta,
    model: ChatModel,
    data: AnswerTaskData,
    system_kwargs: dict[str, Any],
    consumed: set[str],
) -> AnswerSystem:
    """Instantiate a system with the kwargs its constructor accepts."""
    if meta.loader is None:
        raise ValueError(f"System {meta.name!r} has no in-process loader.")
    params = inspect.signature(meta.loader).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    used = {k for k in system_kwargs if accepts_any or k in params} - _HARBOR_KWARGS
    consumed |= used
    kwargs = {k: system_kwargs[k] for k in used}
    if meta.needs_gold:
        # Gold docs keyed by question text, since answer() receives only the question.
        kwargs.setdefault(
            "gold",
            {
                data.questions[qid]: ids
                for qid, ids in data.gold_by_qid.items()
                if qid in data.questions
            },
        )
    return meta.load(model, **kwargs)


def _run_harbor_batch(
    meta: SystemMeta,
    data: AnswerTaskData,
    questions: dict[str, str],
    references: dict[str, str],
    *,
    model: ChatModel | str | None,
    judge: Judge,
    work_dir: str | Path | None,
    **kwargs: Any,
) -> AnswerEvaluationResult:
    # One batch Harbor job over all questions (Harbor's canonical adapter pattern),
    # then score the collected answers with the same Judge as every other system.
    if shutil.which("harbor") is None:
        raise ImportError(
            "Harbor is required for containerized agents. Install it with "
            'pip install "mteb[agentic-agents]" and a running Docker daemon.'
        )
    if meta.harbor_agent is None:
        raise ValueError(f"Harbor system {meta.name!r} has no Harbor agent id.")
    if model is None:
        raise ValueError("Pass model= a model name or ChatModel for Harbor systems.")
    base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="mteb-harbor-"))
    # A shared corpus is materialized once and bind-mounted; per-question corpora
    # (documents_by_qid) are baked into each task instead.
    shared = data.documents if data.documents_by_qid is None else None
    mount = to_harbor_dataset(
        questions,
        data.corpus_for,
        base / "dataset",
        shared_documents=shared,
        agent_timeout_s=kwargs.get("agent_timeout_s", 1800.0),
        retriever_tool=kwargs.get("agent_retriever", False),
    )
    run_harbor(
        base / "dataset",
        meta.harbor_agent,
        model.name if not isinstance(model, str) else model,
        base / "jobs",
        n_concurrent=kwargs.get("n_concurrent", 8),
        agent_env=kwargs.get("agent_env"),
        mount_corpus=mount,
    )
    answers = read_harbor_answers(base / "jobs")  # keyed by q0..qN slug
    metrics = read_harbor_metrics(base / "jobs")  # tokens, cost, latency per trial

    results, correctness, per_question = [], [], []
    for index, (qid, question) in enumerate(questions.items()):
        answer = answers.get(f"q{index}", "")
        score = judge.score(question, answer, references[qid]) if answer else 0.0
        m = metrics.get(f"q{index}", {})
        usage = Usage(
            prompt_tokens=m.get("prompt_tokens", 0),
            completion_tokens=m.get("completion_tokens", 0),
            cost_usd=m.get("cost_usd"),
            latency_s=m.get("latency_s"),
        )
        result = AnswerResult(answer=answer, usage=usage)
        results.append(result)
        correctness.append(score)
        per_question.append(
            _question_record(
                qid,
                result,
                score,
                error=None if answer else "no answer produced",
                gold=data.gold_by_qid.get(qid),
            )
        )
    scores = _finalize(results, correctness, per_question)
    return AnswerEvaluationResult(scores=scores, per_question=per_question)
