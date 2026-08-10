"""Offline tests for the LLM retriever wrappers (query rewrite, HyDE, rerank).

A FakeChatModel and a FakeSearchModel exercise the wrappers without network
access; one test wraps the real BM25 model to check the integration.
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import (
    ChatModelProtocol,
    HyDERetriever,
    QueryRewriteRetriever,
    RerankRetriever,
    SearchProtocol,
)
from mteb.models.chat_models import ChatResponse

CORPUS = Dataset.from_list(
    [
        {"id": "d1", "title": "", "text": "Paris is the capital of France."},
        {"id": "d2", "title": "", "text": "Berlin is the capital of Germany."},
        {
            "id": "d3",
            "title": "",
            "text": "The Seine flows through Paris, the French capital.",
        },
    ]
)

TASK_METADATA = TaskMetadata(
    dataset={"path": "mteb/test", "revision": "main"},
    name="LLMRetrieverTest",
    description="Fixture task for LLM retriever tests.",
    type="Retrieval",
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
)

_KWARGS = {
    "task_metadata": TASK_METADATA,
    "hf_split": "test",
    "hf_subset": "default",
    "encode_kwargs": {},
    "num_proc": None,
}


class FakeChatModel:
    """Deterministic ChatModel: replies from a scripted queue or echoes."""

    def __init__(self, scripted: list[str] | None = None) -> None:
        self._scripted = list(scripted or [])

    def generate(self, messages, **kwargs) -> ChatResponse:
        if self._scripted:
            return ChatResponse(text=self._scripted.pop(0))
        return ChatResponse(text=messages[-1]["content"])


class FakeSearchModel:
    """Minimal SearchProtocol: word-overlap ranking, no encoding or network."""

    def index(self, corpus, **kwargs) -> None:
        self._docs = {row["id"]: row["text"] for row in corpus}

    def search(self, queries, *, top_k, **kwargs):
        out = {}
        for row in queries:
            terms = set(row["text"].lower().split())
            scored = {
                d: float(sum(t in text.lower() for t in terms))
                for d, text in self._docs.items()
            }
            ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
            out[row["id"]] = dict(ranked)
        return out


def _top_ids(retriever, query: str, top_k: int = 2) -> list[str]:
    retriever.index(CORPUS, **_KWARGS)
    out = retriever.search(
        Dataset.from_list([{"id": "q", "text": query}]), top_k=top_k, **_KWARGS
    )
    ranking = out["q"]
    return sorted(ranking, key=lambda d: -ranking[d])


def test_wrappers_implement_search_protocol():
    model = FakeChatModel()
    for retriever in (
        QueryRewriteRetriever(FakeSearchModel(), model),
        HyDERetriever(FakeSearchModel(), model),
        RerankRetriever(FakeSearchModel(), model),
    ):
        assert isinstance(retriever, SearchProtocol)
    assert isinstance(model, ChatModelProtocol)


def test_query_rewrite_searches_on_rewritten_text():
    retriever = QueryRewriteRetriever(
        FakeSearchModel(), FakeChatModel(["capital France"])
    )
    assert _top_ids(retriever, "a vague question")[0] in {"d1", "d3"}


def test_hyde_searches_on_hypothetical_passage():
    retriever = HyDERetriever(
        FakeSearchModel(), FakeChatModel(["Paris is the capital of France"])
    )
    assert _top_ids(retriever, "q")[0] in {"d1", "d3"}


def test_rerank_honors_llm_order():
    retriever = RerankRetriever(
        FakeSearchModel(), FakeChatModel(['["d3", "d1"]']), pool_size=3
    )
    assert _top_ids(retriever, "capital of France") == ["d3", "d1"]


def test_rerank_falls_back_to_base_score_order():
    retriever = RerankRetriever(
        FakeSearchModel(), FakeChatModel(["not json at all"]), pool_size=3
    )
    # d1 outscores the rest on word overlap; the unparseable reply keeps it first.
    ids = _top_ids(retriever, "capital of France paris", top_k=3)
    assert ids[0] == "d1" and len(ids) == 3


def test_wrappers_compose():
    # Rerank over a query-rewriting base: wrappers are SearchProtocol, so they stack.
    retriever = RerankRetriever(
        QueryRewriteRetriever(FakeSearchModel(), FakeChatModel(["capital France"])),
        FakeChatModel(['["d1"]']),
        pool_size=3,
    )
    assert _top_ids(retriever, "a vague question", top_k=1) == ["d1"]


def test_query_rewrite_over_real_bm25():
    import mteb

    retriever = QueryRewriteRetriever(
        mteb.get_model("mteb/baseline-bb25"), FakeChatModel(["capital France"])
    )
    assert _top_ids(retriever, "a vague question")[0] in {"d1", "d3"}


def test_litellm_chat_model_offline():
    # LiteLLM's mock_response skips the network; cost comes from its local table.
    pytest.importorskip("litellm")
    from mteb.models import LiteLLMChatModel

    model = LiteLLMChatModel("gpt-4o")
    assert isinstance(model, ChatModelProtocol)
    out = model.generate(
        [{"role": "user", "content": "capital?"}], mock_response="Paris"
    )
    assert out.text == "Paris"
    assert out.cost_usd is not None and out.cost_usd > 0  # priced from local table
