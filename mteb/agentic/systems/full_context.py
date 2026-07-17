"""Full-context (long-context) paradigm: read the whole corpus, no retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import add_usage

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel, CorpusHandle

_PROMPT = (
    "Answer the question concisely using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
_WINDOW_PROMPT = (
    "This is one part of a longer context. Using only what is below, answer the "
    'question if the answer is present, otherwise reply exactly "NOT FOUND".\n\n'
    "Context part:\n{context}\n\nQuestion: {question}"
)
_AGGREGATE_PROMPT = (
    "Below are answers from different parts of a long context. Combine them into "
    'one concise final answer to the question, ignoring any "NOT FOUND" replies.\n\n'
    "Partial answers:\n{partials}\n\nQuestion: {question}"
)


class FullContextSystem:
    """Put the entire corpus in the prompt and answer, with no retriever.

    If the corpus exceeds max_context_chars, questions are reported as not
    applicable rather than answered from a truncated corpus. Set
    max_context_chars to roughly four times the model's token window.
    """

    def __init__(self, model: ChatModel, *, max_context_chars: int = 500_000) -> None:
        self.model = model
        self.max_context_chars = max_context_chars
        self.name = f"full-context/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer from the whole corpus, or report not applicable if it does not fit."""
        # Size check before concatenating: oversized corpora short-circuit to N/A.
        total = sum(len(doc.get("text", "")) for doc in corpus.documents.values())
        if total > self.max_context_chars:
            return AnswerResult(answer="", applicable=False)
        context = "\n\n".join(
            f"[{d}] {doc.get('text', '')}" for d, doc in corpus.documents.items()
        )
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _PROMPT.format(context=context, question=question),
                }
            ]
        )
        # No cited docs: nothing was selected, the whole corpus was read.
        return AnswerResult(
            answer=out.text,
            usage=Usage(
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                num_llm_calls=1,
                cost_usd=out.cost_usd,
            ),
        )


class WindowedFullContextSystem:
    """Long-context via a sliding window over the corpus, no retriever.

    Unlike FullContextSystem (which reports N/A when the corpus exceeds the
    window), this always applies: it streams the corpus into overlapping windows,
    answers each, then has the model aggregate a final answer. Windows are capped
    at max_windows so cost stays bounded on huge corpora (only a prefix is read,
    noted in the trace). Mirrors the windowed LLM full-context baseline of Cao et
    al. (2026), "Coding Agents are Effective Long-Context Processors".
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        window_chars: int = 800_000,
        overlap_chars: int = 200_000,
        max_windows: int = 8,
    ) -> None:
        self.model = model
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.max_windows = max_windows
        self.name = f"windowed-full-context/{model.name}"

    def _gather_text(self, corpus: CorpusHandle) -> str:
        """Concatenate the context, bounded to what max_windows can cover (avoids OOM)."""
        cap = self.max_windows * self.window_chars
        parts: list[str] = []
        total = 0
        for doc_id, doc in corpus.documents.items():
            piece = f"[{doc_id}] {doc.get('text', '')}"
            parts.append(piece)
            total += len(piece) + 2
            if total >= cap:
                break
        return "\n\n".join(parts)

    def _windows(self, corpus: CorpusHandle) -> tuple[list[str], bool]:
        """Split the context into overlapping character windows (sliding window)."""
        text = self._gather_text(corpus)
        if len(text) <= self.window_chars:
            return [text], False
        stride = max(1, self.window_chars - self.overlap_chars)
        windows: list[str] = []
        pos = 0
        while pos < len(text) and len(windows) < self.max_windows:
            windows.append(text[pos : pos + self.window_chars])
            pos += stride
        return windows, pos < len(text)

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer over sliding windows of the corpus, aggregating per-window answers."""
        windows, truncated = self._windows(corpus)
        usage = Usage()
        partials: list[str] = []
        for window in windows:
            out = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _WINDOW_PROMPT.format(
                            context=window, question=question
                        ),
                    }
                ]
            )
            add_usage(usage, out)
            partials.append(out.text)
        if len(partials) == 1:
            answer = partials[0]
        else:
            agg = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _AGGREGATE_PROMPT.format(
                            partials="\n".join(f"- {p}" for p in partials),
                            question=question,
                        ),
                    }
                ]
            )
            add_usage(usage, agg)
            answer = agg.text
        return AnswerResult(
            answer=answer,
            usage=usage,
            trace=[{"windows": len(windows), "truncated": truncated}],
        )
