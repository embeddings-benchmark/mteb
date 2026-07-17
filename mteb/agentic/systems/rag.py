"""One-shot RAG paradigm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import add_usage, join_context

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel, CorpusHandle

_RAG_PROMPT = (
    "Answer the question concisely using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


class RetrieveThenAnswer:
    """One-shot RAG: retrieve top-k once, answer from the retrieved context."""

    def __init__(
        self, model: ChatModel, *, top_k: int = 5, snippet_chars: int | None = None
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.name = f"rag/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Retrieve once then answer from the retrieved context."""
        hits = corpus.search(question, top_k=self.top_k)
        doc_ids = [doc_id for doc_id, _ in hits]
        context = join_context(corpus, doc_ids, snippet_chars=self.snippet_chars)
        usage = Usage(num_tool_calls=1)
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _RAG_PROMPT.format(context=context, question=question),
                }
            ]
        )
        add_usage(usage, out)
        return AnswerResult(answer=out.text, cited_doc_ids=doc_ids, usage=usage)
