"""Answer-mode system registry, mirroring mteb.get_model / ModelMeta.

To add a system, create a module implementing AnswerSystem and list a SystemMeta
in SYSTEM_REGISTRY. In-process systems declare a loader; Harbor agents declare
kind="harbor" and an agent id, and run as a batch job (see evaluate).

corpus_kind is the corpus access an in-process system needs: "memory" (raw
documents), "retrieval" (a first-stage retriever), or "files".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mteb.agentic.systems.baselines import ClosedBookSystem, OracleContextSystem
from mteb.agentic.systems.full_context import (
    FullContextSystem,
    WindowedFullContextSystem,
)
from mteb.agentic.systems.iterative_rag import IterativeRAG
from mteb.agentic.systems.rag import RetrieveThenAnswer
from mteb.agentic.systems.rlm import RLMSystem
from mteb.agentic.systems.search_agent import SearchAgent

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.agentic.interface import AnswerSystem


@dataclass
class SystemMeta:
    """Metadata for one answer-mode system (peer of ModelMeta)."""

    name: str
    description: str
    loader: Callable[..., AnswerSystem] | None = None  # in-process systems
    corpus_kind: str = "memory"  # memory, retrieval, or files
    kind: str = "in-process"  # in-process or harbor
    harbor_agent: str | None = None  # `harbor run -a` id, for kind="harbor"

    def load(self, *args: Any, **kwargs: Any) -> AnswerSystem:
        """Instantiate the in-process system; first arg is the ChatModel."""
        if self.loader is None:
            raise TypeError(
                f"System {self.name!r} is a {self.kind} agent; run it through "
                "mteb.agentic.evaluate."
            )
        return self.loader(*args, **kwargs)


def _harbor(name: str) -> SystemMeta:
    return SystemMeta(
        name,
        f"{name} agent in a Harbor container over the corpus files, no retriever.",
        kind="harbor",
        harbor_agent=name,
    )


SYSTEM_REGISTRY: dict[str, SystemMeta] = {
    meta.name: meta
    for meta in [
        SystemMeta(
            "closed-book",
            "Floor: answer from parametric memory, ignore the corpus.",
            ClosedBookSystem,
        ),
        SystemMeta(
            "full-context",
            "Long-context: put the whole corpus in the prompt, no retriever "
            "(N/A when the corpus exceeds the window).",
            FullContextSystem,
        ),
        SystemMeta(
            "windowed-full-context",
            "Long-context via a sliding window over the corpus with per-window "
            "answers aggregated; always applies (bounded by max_windows).",
            WindowedFullContextSystem,
        ),
        SystemMeta(
            "rag",
            "One-shot: retrieve top-k, then answer.",
            RetrieveThenAnswer,
            corpus_kind="retrieval",
        ),
        SystemMeta(
            "iterative-rag",
            "Decompose-retrieve-reason loop (Self-Ask/IRCoT): retrieve per "
            "sub-query until ready, then answer.",
            IterativeRAG,
            corpus_kind="retrieval",
        ),
        SystemMeta(
            "search-agent",
            "Iterative reason-act over search and get_document tools "
            "(BrowseComp-Plus style).",
            SearchAgent,
            corpus_kind="retrieval",
        ),
        SystemMeta(
            "oracle",
            "Ceiling: answer from gold documents (gold wired from the task).",
            OracleContextSystem,
        ),
        SystemMeta(
            "rlm",
            "Recursive Language Model over the raw corpus: in-process orchestrator "
            "that writes code to search it (local execution by default; docker or a "
            "cloud backend optional), no retriever.",
            RLMSystem,
        ),
        _harbor("claude-code"),
        _harbor("codex"),
        _harbor("mini-swe-agent"),
        _harbor("openhands"),
        _harbor("hermes"),
    ]
}


def list_systems() -> list[str]:
    """Names of all registered answer-mode systems."""
    return sorted(SYSTEM_REGISTRY)


def get_system_meta(name: str) -> SystemMeta:
    """Fetch a system's metadata by name."""
    if name not in SYSTEM_REGISTRY:
        raise KeyError(f"Unknown system {name!r}. Available: {list_systems()}")
    return SYSTEM_REGISTRY[name]


def get_system(name: str, *args: Any, **kwargs: Any) -> AnswerSystem:
    """Load a registered in-process system, passing model and scaffold kwargs."""
    return get_system_meta(name).load(*args, **kwargs)


__all__ = [
    "SYSTEM_REGISTRY",
    "ClosedBookSystem",
    "FullContextSystem",
    "IterativeRAG",
    "OracleContextSystem",
    "RLMSystem",
    "RetrieveThenAnswer",
    "SearchAgent",
    "SystemMeta",
    "WindowedFullContextSystem",
    "get_system",
    "get_system_meta",
    "list_systems",
]
