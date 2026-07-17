"""Iterative / decomposition RAG paradigm (Self-Ask / IRCoT style).

Interleaves retrieval and reasoning: the model proposes a follow-up sub-query,
that query is retrieved, the evidence is accumulated, and the loop repeats until
the model is ready to answer. The strong multi-hop baseline that plain one-shot
RAG is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import add_usage, join_context

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel, CorpusHandle

_FOLLOWUP = (
    "You are answering a multi-hop question by searching a corpus one step at a "
    "time. Given the question and the evidence gathered so far, reply with the "
    "single most useful next search query. If the evidence is already enough to "
    'answer, reply exactly "READY".\n\n'
    "Question: {question}\n\nEvidence so far:\n{evidence}"
)
_ANSWER = (
    "Answer the question concisely using only the evidence below.\n\n"
    "Evidence:\n{evidence}\n\nQuestion: {question}"
)


class IterativeRAG:
    """Decompose-retrieve-reason loop: retrieve per sub-query until ready, then answer."""

    def __init__(
        self,
        model: ChatModel,
        *,
        top_k: int = 3,
        max_hops: int = 4,
        snippet_chars: int | None = None,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.max_hops = max_hops
        self.snippet_chars = snippet_chars
        self.name = f"iterative-rag/{model.name}"

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Iteratively retrieve for model-proposed sub-queries, then answer."""
        usage = Usage()
        doc_ids: list[str] = []
        blocks: list[str] = []
        queries: list[str] = []
        for _ in range(self.max_hops):
            evidence = "\n\n".join(blocks) if blocks else "(none yet)"
            follow = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _FOLLOWUP.format(
                            question=question, evidence=evidence
                        ),
                    }
                ]
            )
            add_usage(usage, follow)
            query = follow.text.strip()
            if not query or query.upper().startswith("READY"):
                break
            queries.append(query)
            hits = corpus.search(query, top_k=self.top_k)
            usage.num_tool_calls += 1
            new_ids = [doc_id for doc_id, _ in hits if doc_id not in doc_ids]
            doc_ids.extend(new_ids)
            if new_ids:
                blocks.append(
                    join_context(corpus, new_ids, snippet_chars=self.snippet_chars)
                )
        evidence = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _ANSWER.format(evidence=evidence, question=question),
                }
            ]
        )
        add_usage(usage, out)
        return AnswerResult(
            answer=out.text,
            cited_doc_ids=doc_ids,
            usage=usage,
            trace=[{"sub_queries": queries}],
        )
