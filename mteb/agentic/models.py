"""ChatModel implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.interface import ChatResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.agentic.interface import Message


class OpenAIChatModel:
    """ChatModel for any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM)."""

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
        from openai import OpenAI

        self.name = model
        # Exposed so external agents (RLM, Harbor) can point at the same endpoint.
        self.base_url = base_url
        self.api_key = api_key
        # Client-level timeout and retries so one stuck request cannot stall a run.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._kwargs = kwargs

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Call the chat completions endpoint. Pass tools to enable tool calling."""
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=list(messages),
            **{**self._kwargs, **kwargs},
        )
        message = resp.choices[0].message
        usage = resp.usage
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (message.tool_calls or [])
        ]
        return ChatResponse(
            text=message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            tool_calls=tool_calls,
        )
