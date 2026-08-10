"""Chat model interface and LiteLLM client for LLM-augmented retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    """One chat completion with usage accounting."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class ChatModelProtocol(Protocol):
    """Interface for chat models used by LLM retrievers."""

    def generate(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        """Run one chat completion over the messages.

        Args:
            messages: Chat messages, each a {"role", "content"} mapping.
            **kwargs: Provider arguments, e.g. tools= for tool calling.

        Returns:
            The completion with token and cost accounting.
        """
        ...


class LiteLLMChatModel:
    """ChatModelProtocol over LiteLLM: any provider, built-in cost accounting.

    LiteLLM ships a local per-model pricing table, so cost_usd is populated
    for hosted models with no extra wiring; models absent from the table
    (local endpoints) report None. Use provider-prefixed names, e.g.
    "openai/Qwen3-32B" for a vLLM or other OpenAI-compatible server.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LiteLLMChatModel requires the litellm package. Install it "
                "with pip install litellm."
            ) from exc
        self.name = model
        self.base_url = base_url
        self.api_key = api_key
        self._kwargs = {"timeout": timeout, "num_retries": max_retries, **kwargs}

    def generate(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        """Run one chat completion. Pass tools= to enable tool calling."""
        import litellm

        resp = litellm.completion(
            model=self.name,
            messages=list(messages),
            api_base=self.base_url,
            api_key=self.api_key,
            **{**self._kwargs, **kwargs},
        )
        message = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (message.tool_calls or [])
        ]
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None  # model absent from the pricing table (local endpoints)
        return ChatResponse(
            text=message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            tool_calls=tool_calls,
        )
