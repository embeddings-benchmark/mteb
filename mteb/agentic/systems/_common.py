"""Shared helpers for answer-mode systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.agentic.interface import ChatResponse, CorpusHandle, Usage


def join_context(
    corpus: CorpusHandle,
    doc_ids: Iterable[str],
    *,
    snippet_chars: int | None = None,
) -> str:
    """Join document texts into one context block, truncating each to snippet_chars.

    snippet_chars bounds per-document length so many or long documents stay within
    the model context window. None keeps full texts.
    """
    parts = []
    for doc_id in doc_ids:
        text = corpus.get(doc_id).get("text", "")
        parts.append(text[:snippet_chars] if snippet_chars is not None else text)
    return "\n\n".join(parts)


def add_usage(usage: Usage, response: ChatResponse) -> None:
    """Fold one chat response into a running usage record."""
    usage.prompt_tokens += response.prompt_tokens
    usage.completion_tokens += response.completion_tokens
    usage.num_llm_calls += 1
    if response.cost_usd is not None:
        usage.cost_usd = (usage.cost_usd or 0.0) + response.cost_usd
