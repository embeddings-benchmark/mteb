"""Core contract for answer-mode retrieval systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

# Provider agnostic chat message, e.g. {"role": "user", "content": "..."}.
# Assistant messages may also carry tool_calls, tool messages a tool_call_id.
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation the model requested. arguments is a raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    """Result of one chat completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class ChatModel(Protocol):
    """Provider agnostic chat interface.

    In-process systems call generate; external agents read name, base_url, and
    api_key to target the same endpoint themselves.
    """

    name: str
    base_url: str | None
    api_key: str | None

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Generate a single completion for a chat transcript."""
        ...


@runtime_checkable
class CorpusHandle(Protocol):
    """Read and search access to a fixed corpus for in-process systems.

    Retrieval systems use get + search (search is the first-stage retriever).
    Raw-access systems (RLM, grep agents) read the whole corpus via documents,
    with no first stage.
    """

    @property
    def documents(self) -> dict[str, dict[str, str]]:
        """All documents, for systems that read the whole corpus with no retriever."""
        ...

    def get(self, doc_id: str) -> dict[str, str]:
        """Fetch one document as a mapping with at least id and text."""
        ...

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve candidate documents as (doc_id, score) pairs (first-stage retriever)."""
        ...


@dataclass
class Usage:
    """Cost and latency accounting for one answer."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    cost_usd: float | None = None
    latency_s: float | None = None


@dataclass
class AnswerResult:
    """What a system returns for one question."""

    answer: str
    cited_doc_ids: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    trace: list[dict[str, Any]] = field(default_factory=list)
    # False when the paradigm cannot run this task (e.g. the corpus exceeds the
    # window for full-context). Excluded from accuracy, surfaced as coverage.
    applicable: bool = True


@runtime_checkable
class AnswerSystem(Protocol):
    """End to end retrieval system that produces an answer, not a ranking.

    Systems receive only the question and the corpus handle, never the qrels.
    """

    name: str

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer a single question using the corpus."""
        ...
