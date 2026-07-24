"""Floor and ceiling baselines that bracket every system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.agentic.interface import AnswerResult, Usage

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
    """Ceiling baseline. Answers from gold documents passed at construction."""

    def __init__(self, model: ChatModel, gold: Mapping[str, list[str]]) -> None:
        self.model = model
        self.gold = gold
        self.name = f"oracle-context/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Answer from the gold documents configured for this question."""
        doc_ids = self.gold.get(question, [])
        docs = [corpus.get(doc_id).get("text", "") for doc_id in doc_ids]
        context = "\n\n".join(docs)
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
