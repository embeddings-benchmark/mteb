"""Floor and ceiling baselines: parametric memory only, and gold documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import join_context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mteb.agentic.interface import ChatModel, CorpusHandle

_ANSWER_PROMPT = "Answer the question concisely.\n\nQuestion: {question}"
_CONTEXT_PROMPT = (
    "Answer the question concisely using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


class ClosedBookSystem:
    """Floor baseline. Answers from parametric memory, ignores the corpus."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.name = f"closed-book/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer from parametric memory, ignoring the corpus."""
        out = self.model.generate(
            [{"role": "user", "content": _ANSWER_PROMPT.format(question=question)}]
        )
        return AnswerResult(
            answer=out.text,
            usage=Usage(
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                num_llm_calls=1,
                cost_usd=out.cost_usd,
            ),
        )


class OracleContextSystem:
    """Ceiling baseline. Answers from gold documents handed in at construction.

    Gold documents are keyed by question text, so answer() still receives only
    the question.
    """

    def __init__(
        self,
        model: ChatModel,
        gold: Mapping[str, list[str]],
        *,
        snippet_chars: int | None = None,
    ) -> None:
        self.model = model
        self.gold = gold
        self.snippet_chars = snippet_chars
        self.name = f"oracle-context/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer from the gold documents configured for this question."""
        doc_ids = self.gold.get(question, [])
        context = join_context(corpus, doc_ids, snippet_chars=self.snippet_chars)
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _CONTEXT_PROMPT.format(
                        context=context, question=question
                    ),
                }
            ]
        )
        return AnswerResult(
            answer=out.text,
            cited_doc_ids=list(doc_ids),
            usage=Usage(
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                num_llm_calls=1,
                cost_usd=out.cost_usd,
            ),
        )
