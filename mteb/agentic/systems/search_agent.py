"""Search-agent paradigm: a deep-research agent with a retriever tool.

This mirrors the BrowseComp-Plus reference setup: the model is given a search tool
(top-k hits with truncated snippets) and a get_document tool, and it calls them
in an interleaved reason-act loop until it produces a final answer. Retrieval is
backed by the CorpusHandle, so any retriever (BM25, dense) plugs in unchanged.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import add_usage

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel, CorpusHandle

# Prompt verbatim from BrowseComp-Plus (search_agent/prompts.py, QUERY_TEMPLATE).
_QUERY_TEMPLATE = (
    "You are a deep research agent. You need to answer the given question by "
    "interacting with a search engine, using the search and get_document tools "
    "provided. Please perform reasoning and use the tools step by step, in an "
    "interleaved manner. You may use the search and get_document tools multiple "
    "times.\n\nQuestion: {question}\n\nYour response should be in the following "
    "format:\nExplanation: {{your explanation for your final answer. cite evidence "
    "docids in square brackets, e.g. [20].}}\nExact Answer: {{your succinct, final "
    "answer}}\nConfidence: {{your confidence score between 0% and 100%}}"
)
_EXACT_ANSWER = re.compile(r"Exact Answer:\s*(.+?)(?:\nConfidence:|\Z)", re.DOTALL)


def _final_answer(text: str) -> str:
    match = _EXACT_ANSWER.search(text)
    return match.group(1).strip() if match else text.strip()


class SearchAgent:
    """Reason-act agent over search + get_document tools backed by the corpus."""

    def __init__(
        self,
        model: ChatModel,
        *,
        k: int = 5,
        snippet_chars: int = 2000,
        max_iterations: int = 30,
    ) -> None:
        self.model = model
        self.k = k
        self.snippet_chars = snippet_chars  # matches the BCP 512-token snippet cap
        self.max_iterations = max_iterations
        self.name = f"search-agent/{model.name}"

    def _tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        f"Search the corpus. Returns top-{self.k} hits with docid, "
                        "score, and a snippet of the document."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": "Retrieve the full text of a document by its docid.",
                    "parameters": {
                        "type": "object",
                        "properties": {"docid": {"type": "string"}},
                        "required": ["docid"],
                    },
                },
            },
        ]

    def _run_tool(
        self, name: str, args: dict, corpus: CorpusHandle, seen: set[str]
    ) -> object:
        if name == "search":
            hits = corpus.search(args.get("query", ""), top_k=self.k)
            results = []
            for doc_id, score in hits:
                seen.add(doc_id)
                text = corpus.get(doc_id).get("text", "")[: self.snippet_chars]
                results.append({"docid": doc_id, "score": score, "snippet": text})
            return results
        if name == "get_document":
            doc_id = args.get("docid", "")
            seen.add(doc_id)
            return {"docid": doc_id, "text": corpus.get(doc_id).get("text", "")}
        return {"error": f"unknown tool {name}"}

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Run the reason-act loop, then return the extracted final answer."""
        usage = Usage()
        seen: set[str] = set()
        trace: list[dict[str, Any]] = []
        tools = self._tools()
        messages: list[dict] = [
            {"role": "user", "content": _QUERY_TEMPLATE.format(question=question)}
        ]
        text = ""
        for _ in range(self.max_iterations):
            out = self.model.generate(messages, tools=tools, tool_choice="auto")
            add_usage(usage, out)
            text = out.text
            if not out.tool_calls:
                break
            # One trace step per reasoning turn: the tools it chose to call.
            trace.append(
                {
                    "tool_calls": [
                        {"name": tc.name, "args": tc.arguments} for tc in out.tool_calls
                    ]
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": out.text or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in out.tool_calls
                    ],
                }
            )
            for tc in out.tool_calls:
                usage.num_tool_calls += 1
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(tc.name, args, corpus, seen)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
        else:
            # Out of iterations still calling tools: force a final answer.
            messages.append(
                {"role": "user", "content": "Provide your final answer now."}
            )
            out = self.model.generate(messages)
            add_usage(usage, out)
            text = out.text
        return AnswerResult(
            answer=_final_answer(text),
            cited_doc_ids=sorted(seen),
            usage=usage,
            trace=trace,
        )
