"""Chat model interface and LiteLLM client for LLM-augmented retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ChatResponse:
    """One chat completion with token and cost accounting."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None  # None when the model has no pricing entry


@runtime_checkable
class ChatModelProtocol(Protocol):
    """Interface for chat models used by LLM retrievers."""

    def generate(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        """Run one chat completion over the messages.

        Args:
            messages: Chat messages, each a {"role", "content"} mapping.
            **kwargs: Extra provider arguments.

        Returns:
            The completion with cost accounting.
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
                'with pip install "mteb[litellm]".'
            ) from exc
        self.name = model
        self.base_url = base_url
        self.api_key = api_key
        self._kwargs = {"timeout": timeout, "num_retries": max_retries, **kwargs}

    def generate(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        """Run one chat completion."""
        import litellm

        resp = litellm.completion(
            model=self.name,
            messages=list(messages),
            api_base=self.base_url,
            api_key=self.api_key,
            **{**self._kwargs, **kwargs},
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None  # model absent from the pricing table (local endpoints)
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
        )
