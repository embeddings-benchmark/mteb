"""Core contract for answer-mode retrieval systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

# Provider-agnostic chat message.
Message = dict[str, str]


@dataclass
class ChatResponse:
    """Result of one chat completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None


@runtime_checkable
class ChatModel(Protocol):
    """Provider-agnostic chat interface."""

    name: str

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Generate a single completion for a chat transcript."""
        ...


@runtime_checkable
class CorpusHandle(Protocol):
    """Read access to a fixed corpus."""

    def get(self, doc_id: str) -> dict[str, str]:
        """Fetch one document as a mapping with at least id and text."""
        ...


@dataclass
class Usage:
    """Cost and latency accounting for one answer."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_llm_calls: int = 0
    cost_usd: float | None = None
    latency_s: float | None = None


@dataclass
class AnswerResult:
    """What a system returns for one question."""

    answer: str
    cited_doc_ids: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class AnswerSystem(Protocol):
    """End-to-end system that produces an answer, not a ranking."""

    name: str

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer a single question using the corpus."""
        ...
