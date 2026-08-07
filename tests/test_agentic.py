"""Fast, offline tests for the answer-mode retrieval benchmark (mteb.agentic).

A FakeChatModel and a FakeSearchModel (a tiny SearchProtocol) exercise the
contract, systems, registries, evaluator, and metrics without network access.
One test wraps MTEB's real BM25 to check the integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mteb.agentic import (
    AnswerEvaluator,
    ChatResponse,
    ClosedBookSystem,
    ExactMatchJudge,
    LLMJudge,
    RetrievalCorpus,
    RetrieveThenAnswer,
    SearchAgent,
    ToolCall,
    evaluate,
    from_mteb_retrieval,
    from_per_question,
    get_system,
    list_systems,
    list_tasks,
)
from mteb.agentic.corpus import InMemoryCorpus
from mteb.agentic.systems import IterativeRAG, WindowedFullContextSystem
from mteb.agentic.systems._common import join_context

CORPUS = {
    "d1": {"text": "Paris is the capital of France."},
    "d2": {"text": "Berlin is the capital of Germany."},
    "d3": {"text": "The Seine flows through Paris, the French capital."},
}


class FakeChatModel:
    """Deterministic ChatModel: replies from a scripted queue or echoes."""

    name = "fake"
    base_url = None
    api_key = None

    def __init__(self, scripted: list[ChatResponse] | None = None) -> None:
        self._scripted = list(scripted or [])

    def generate(self, messages, **kwargs) -> ChatResponse:
        if self._scripted:
            return self._scripted.pop(0)
        return ChatResponse(text="Paris")


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


def corpus() -> RetrievalCorpus:
    return RetrievalCorpus(CORPUS, FakeSearchModel())


def test_retrieval_corpus_search_and_get():
    c = corpus()
    hits = c.search("capital of France", top_k=2)
    assert hits[0][0] in {"d1", "d3"}
    assert c.get("d1")["text"].startswith("Paris")


def test_retrieval_corpus_with_mteb_bm25():
    import mteb

    c = RetrievalCorpus(CORPUS, mteb.get_model("mteb/baseline-bb25"))
    hits = c.search("capital of France", top_k=2)
    assert hits and hits[0][0] in {"d1", "d3"}


def test_join_context_truncates_per_doc():
    c = corpus()
    assert join_context(c, ["d1"]).startswith("Paris")
    assert len(join_context(c, ["d1"], snippet_chars=5)) == 5


def test_query_rewrite_retriever():
    from mteb.agentic import QueryRewriteRetriever

    # The LLM rewrites the query; the base retriever searches on the rewrite.
    base = QueryRewriteRetriever(
        FakeSearchModel(), FakeChatModel([ChatResponse(text="capital France")])
    )
    hits = RetrievalCorpus(CORPUS, base).search("a vague question", top_k=2)
    assert hits and hits[0][0] in {"d1", "d3"}


def test_hyde_retriever():
    from mteb.agentic import HyDERetriever

    base = HyDERetriever(
        FakeSearchModel(),
        FakeChatModel([ChatResponse(text="Paris is the capital of France")]),
    )
    hits = RetrievalCorpus(CORPUS, base).search("q", top_k=2)
    assert hits and hits[0][0] in {"d1", "d3"}


def test_rerank_retriever_honors_llm_order():
    from mteb.agentic import RerankRetriever

    base = RerankRetriever(
        FakeSearchModel(),
        FakeChatModel([ChatResponse(text='["d3", "d1"]')]),
        pool_size=3,
    )
    hits = RetrievalCorpus(CORPUS, base).search("capital of France", top_k=2)
    assert [h[0] for h in hits] == ["d3", "d1"]  # LLM rerank order preserved


def test_rag_with_llm_retriever_end_to_end():
    from mteb.agentic import RerankRetriever

    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    retriever = RerankRetriever(
        FakeSearchModel(), FakeChatModel([ChatResponse(text='["d1"]')]), pool_size=3
    )
    res = evaluate(
        "rag",
        data,
        model=FakeChatModel(),
        judge=ExactMatchJudge(),
        retriever=retriever,
        top_k=1,
    )
    assert res.scores.accuracy == 1.0


def test_registry_lists_and_loads():
    # Exact registry contents, so a silently dropped system fails the suite.
    assert list_systems() == sorted(
        [
            "closed-book",
            "full-context",
            "windowed-full-context",
            "rag",
            "iterative-rag",
            "search-agent",
            "oracle",
            "rlm",
            "claude-code",
            "codex",
            "mini-swe-agent",
            "openhands",
            "hermes",
        ]
    )
    assert isinstance(get_system("closed-book", FakeChatModel()), ClosedBookSystem)
    with pytest.raises(KeyError, match="Did you mean 'rag'"):
        get_system("ragg", FakeChatModel())


def test_search_agent_tool_loop():
    scripted = [
        ChatResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="c1", name="search", arguments='{"query": "capital France"}'
                )
            ],
        ),
        ChatResponse(
            text="Explanation: from the docs [d1].\nExact Answer: Paris\nConfidence: 90%"
        ),
    ]
    agent = SearchAgent(FakeChatModel(scripted), top_k=2)
    result = agent.answer("What is the capital of France?", corpus())
    assert result.answer == "Paris"
    assert result.usage.num_tool_calls == 1
    assert result.cited_doc_ids
    # The reason-act loop records one trace step per tool-calling turn.
    assert result.trace == [
        {"tool_calls": [{"name": "search", "args": '{"query": "capital France"}'}]}
    ]


def test_evaluator_resilient_and_scored():
    class Boom:
        name = "boom"

        def answer(self, question, corpus):  # noqa: PLR6301
            raise RuntimeError("transient")

    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    res = AnswerEvaluator(data.questions, data.references, corpus(), ExactMatchJudge())(
        Boom()
    )
    assert res.scores.accuracy == 0.0
    assert res.per_question[0]["error"] is not None


def test_evaluator_concurrent_matches_sequential():
    questions = {f"q{i}": "capital of France?" for i in range(6)}
    refs = dict.fromkeys(questions, "Paris")
    c = corpus()
    seq = AnswerEvaluator(questions, refs, c, ExactMatchJudge(), max_workers=1)(
        RetrieveThenAnswer(FakeChatModel(), top_k=1)
    )
    conc = AnswerEvaluator(questions, refs, c, ExactMatchJudge(), max_workers=4)(
        RetrieveThenAnswer(FakeChatModel(), top_k=1)
    )
    assert seq.scores.accuracy == conc.scores.accuracy == 1.0
    # Records must line up question-for-question regardless of worker count.
    strip = lambda r: {k: v for k, v in r.items() if k != "latency_s"}  # noqa: E731
    assert [strip(r) for r in seq.per_question] == [strip(r) for r in conc.per_question]


def test_evaluate_door_in_process():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    # closed-book needs no retriever; oracle wires its gold from the task.
    closed = evaluate("closed-book", data, model=FakeChatModel())
    assert closed.scores.n == 1 and closed.scores.accuracy == 1.0
    assert "trace" in closed.per_question[0]  # trace is surfaced, not discarded
    oracle = evaluate("oracle", data, model=FakeChatModel())
    assert oracle.scores.accuracy == 1.0 and oracle.per_question[0]["cited_doc_ids"]


def test_full_context_fits_and_gates():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    # Corpus fits the budget: it answers and coverage is full.
    fits = evaluate("full-context", data, model=FakeChatModel())
    assert fits.scores.accuracy == 1.0 and fits.scores.coverage == 1.0
    # Corpus exceeds the budget: N/A, excluded from accuracy, coverage 0.
    gated = evaluate("full-context", data, model=FakeChatModel(), max_context_chars=5)
    assert gated.scores.coverage == 0.0 and gated.scores.n_applicable == 0
    assert gated.per_question[0]["applicable"] is False
    assert gated.per_question[0]["correct"] is None


def test_per_question_corpus():
    # Each question carries its own corpus; the evaluator routes each to its own.
    from mteb.agentic import AnswerResult

    data = from_per_question(
        corpora={
            "q1": {"a": {"text": "The capital of France is Paris."}},
            "q2": {"b": {"text": "The capital of Japan is Tokyo."}},
        },
        queries={"q1": "capital of France?", "q2": "capital of Japan?"},
        relevant_docs={"q1": {"a": 1}, "q2": {"b": 1}},
        answers={"q1": "Paris", "q2": "Tokyo"},
    )
    seen = {}

    class Peek:
        name = "peek"

        def answer(self, question, corpus):  # noqa: PLR6301
            seen[question] = set(corpus.documents)
            return AnswerResult(answer="")

    res = evaluate(Peek(), data, model=FakeChatModel(), judge=ExactMatchJudge())
    assert res.scores.n == 2
    assert seen["capital of France?"] == {"a"}  # q1 saw only its own corpus
    assert seen["capital of Japan?"] == {"b"}


def test_evaluate_door_retrieval_axis():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    # A retrieval system must be told which retriever to use.
    with pytest.raises(ValueError, match="retriever"):
        evaluate("rag", data, model=FakeChatModel())
    res = evaluate(
        "rag", data, model=FakeChatModel(), retriever=FakeSearchModel(), top_k=1
    )
    assert res.scores.accuracy == 1.0


def test_evaluate_batch_reuses_retrieval_index():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )

    class CountingSearchModel(FakeSearchModel):
        def __init__(self) -> None:
            self.index_calls = 0

        def index(self, corpus, **kwargs) -> None:
            self.index_calls += 1
            super().index(corpus, **kwargs)

    retriever = CountingSearchModel()
    results = evaluate(
        task=data,
        systems=["rag", "iterative-rag", "search-agent"],
        model=FakeChatModel(),
        retriever=retriever,
    )

    assert set(results) == {"rag", "iterative-rag", "search-agent"}
    assert all(result.scores.n == 1 for result in results.values())
    assert retriever.index_calls == 1


def test_evaluate_batch_loads_task_and_model_once(monkeypatch):
    import importlib

    ev = importlib.import_module("mteb.agentic.evaluate")
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    calls = {"task": 0, "model": 0}

    def load_task():
        calls["task"] += 1
        return data

    def fake_task_meta(name):
        assert name == "Tiny"
        from mteb.agentic import TaskMeta

        return TaskMeta("Tiny", "tiny", load_task, default_judge="exact_match")

    def fake_model(name):
        assert name == "fake-model"
        calls["model"] += 1
        return FakeChatModel()

    monkeypatch.setattr(ev, "get_task_meta", fake_task_meta)
    monkeypatch.setattr(ev, "OpenAIChatModel", fake_model)
    results = evaluate(
        task="Tiny",
        systems=["closed-book", "oracle"],
        model="fake-model",
    )

    assert set(results) == {"closed-book", "oracle"}
    assert calls == {"task": 1, "model": 1}


def test_evaluate_requires_exactly_one_system_form():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(task=data, model=FakeChatModel())
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(
            "closed-book",
            data,
            systems=["oracle"],
            model=FakeChatModel(),
        )
    with pytest.raises(ValueError, match="at least one"):
        evaluate(task=data, systems=[], model=FakeChatModel())


def test_llm_judge_robust_to_reasoning():
    yes = LLMJudge(
        FakeChatModel(
            [ChatResponse(text="<think>reference says Paris; matches</think> YES")]
        )
    )
    assert yes.score("capital?", "Paris", "Paris") == 1.0
    no = LLMJudge(FakeChatModel([ChatResponse(text="Reasoning... final: NO")]))
    assert no.score("capital?", "Berlin", "Paris") == 0.0


def test_llm_judge_parses_correct_verdict():
    # BrowseComp-style grader output: "correct: yes|no".
    yes = LLMJudge(FakeChatModel([ChatResponse(text="reasoning\ncorrect: yes")]))
    assert yes.score("q", "Paris", "Paris") == 1.0
    no = LLMJudge(FakeChatModel([ChatResponse(text="extracted...\ncorrect: no")]))
    assert no.score("q", "Berlin", "Paris") == 0.0


def test_official_metric_judges():
    from mteb.agentic import MultipleChoiceJudge, NumericToleranceJudge, QAF1Judge

    f1 = QAF1Judge()
    assert f1.score("q", "Paris", "Paris") == 1.0
    assert 0.0 < f1.score("q", "city of Paris", "Paris") < 1.0
    assert f1.score("q", "Berlin", "Paris") == 0.0

    mcq = MultipleChoiceJudge()
    assert mcq.score("q", "The correct answer is (B).", "(B) Berlin") == 1.0
    assert mcq.score("q", "The correct answer is (A).", "(B) Berlin") == 0.0

    num = NumericToleranceJudge()
    assert num.score("q", "114", "114") == 1.0
    assert num.score("q", "113", "114") == 0.75  # off by one
    assert num.score("q", "incorrect", "incorrect") == 1.0  # label -> exact match


def test_task_default_judge_and_resolution():
    from mteb.agentic import ExactMatchJudge, QAF1Judge, TaskMeta, get_task_meta
    from mteb.agentic.evaluate import _default_judge

    assert get_task_meta("LongBenchV2").default_judge == "mcq"
    assert get_task_meta("HotpotQA").default_judge == "qa_f1"
    assert get_task_meta("BrowseCompPlus").default_judge == "llm"

    qa_meta = TaskMeta("T", "t", lambda: None, default_judge="qa_f1")
    assert isinstance(_default_judge(qa_meta, FakeChatModel()), QAF1Judge)

    # Ad-hoc AnswerTaskData (no TaskMeta) defaults to exact match.
    assert isinstance(_default_judge(None, FakeChatModel()), ExactMatchJudge)

    llm_meta = TaskMeta("T", "t", lambda: None)  # default_judge="llm"
    assert type(_default_judge(llm_meta, FakeChatModel())).__name__ == "LLMJudge"
    # An LLM-judged task must not silently downgrade when no ChatModel is usable.
    with pytest.raises(ValueError, match="LLM judge"):
        _default_judge(llm_meta, "model-name")
    with pytest.raises(ValueError, match="LLM judge"):
        _default_judge(llm_meta, None)


def test_recall_and_calibration_metrics():
    from mteb.agentic.metrics import calibration_error, extract_confidence, recall_at

    assert recall_at(["d1", "d2"], ["d1", "d3"]) == 0.5  # 1 of 2 gold retrieved
    assert recall_at([], ["d1"]) is None  # nothing cited -> N/A, not 0
    assert recall_at(["d1"], []) is None  # no gold -> N/A
    assert recall_at(["d2"], ["d1"]) == 0.0  # cited but missed all gold
    assert extract_confidence("Exact Answer: Paris. Confidence: 90%") == 0.9
    assert extract_confidence("Confidence: 0.9") == 0.9  # fraction form
    assert extract_confidence("no confidence here") is None
    assert calibration_error([1.0, 0.0], [1.0, 0.0]) == 0.0  # perfectly calibrated
    assert calibration_error([1.0], [0.0]) == 1.0  # confident but wrong
    assert calibration_error([None], [1.0]) is None


def test_evaluator_reports_gold_recall():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    res = AnswerEvaluator(
        data.questions,
        data.references,
        corpus(),
        ExactMatchJudge(),
        gold=data.gold_by_qid,
    )(RetrieveThenAnswer(FakeChatModel(), top_k=1))
    assert res.per_question[0]["recall"] == 1.0  # retrieved the gold doc d1
    assert res.scores.mean_recall == 1.0


def test_evaluate_resolves_model_name(monkeypatch):
    # A model name (not a ChatModel) is built into an OpenAIChatModel from env.
    import importlib

    ev = importlib.import_module("mteb.agentic.evaluate")
    built = {}

    def fake_openai(name, **kwargs):
        built["name"] = name
        return FakeChatModel()

    monkeypatch.setattr(ev, "OpenAIChatModel", fake_openai)
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    res = evaluate("closed-book", data, model="my-model")
    assert built["name"] == "my-model" and res.scores.accuracy == 1.0


def test_harbor_batch_dispatch(tmp_path, monkeypatch):
    # Harbor agents run as one batch job: evaluate exports a dataset, runs harbor
    # (mocked here), reads answer artifacts, and scores them like any system.

    data = from_per_question(
        corpora={"q1": {"a": {"text": "Paris is the capital of France."}}},
        queries={"q1": "capital of France?"},
        relevant_docs={"q1": {"a": 1}},
        answers={"q1": "Paris"},
    )
    seen = {}

    def fake_run(dataset_dir, agent, model, jobs_dir, **kwargs):
        seen["agent"] = agent
        toml = (Path(dataset_dir) / "q0" / "task.toml").read_text()  # slug q0
        assert 'name = "mteb-agentic/q0"' in toml  # valid org/name
        art = Path(jobs_dir) / "job" / "q0__abc" / "artifacts"
        art.mkdir(parents=True)
        (art / "answer.txt").write_text("Paris")

    import importlib

    ev_mod = importlib.import_module("mteb.agentic.evaluate")
    monkeypatch.setattr(ev_mod, "run_harbor", fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/harbor")
    res = evaluate(
        "mini-swe-agent", data, model="qwen", judge=ExactMatchJudge(), work_dir=tmp_path
    )
    assert seen["agent"] == "mini-swe-agent"
    assert res.scores.accuracy == 1.0 and res.per_question[0]["answer"] == "Paris"


def test_hotpotqa_transform():
    from mteb.agentic.tasks.hotpotqa import _to_answer_data

    rows = [
        {
            "id": "q1",
            "question": "Which city is the French capital?",
            "answer": "Paris",
            "context": {
                "title": ["France", "Germany"],
                "sentences": [["Paris is the capital.", "It is large."], ["Berlin."]],
            },
            "supporting_facts": {"title": ["France"], "sent_id": [0]},
        }
    ]
    data = _to_answer_data(rows)
    assert data.references["q1"] == "Paris"
    assert data.documents == {}  # original distractor setting: corpus is per-question
    assert (
        data.corpus_for("q1")["France"]["text"] == "Paris is the capital. It is large."
    )
    assert data.gold_by_qid["q1"] == ["France"]


def test_musique_transform():
    from mteb.agentic.tasks.musique import _to_answer_data

    rows = [
        {
            "id": "q1",
            "question": "Spouse of the Green performer?",
            "answer": "Miquette Giraudy",
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "A",
                    "paragraph_text": "distractor",
                    "is_supporting": False,
                },
                {
                    "idx": 1,
                    "title": "B",
                    "paragraph_text": "gold fact",
                    "is_supporting": True,
                },
            ],
        }
    ]
    data = _to_answer_data(rows)
    assert data.references["q1"] == "Miquette Giraudy"
    assert len(data.documents) == 2  # both paragraphs kept, deduped by content
    gold = data.gold_by_qid["q1"]
    assert len(gold) == 1 and data.documents[gold[0]]["text"] == "gold fact"


def test_oolong_transform():
    from mteb.agentic.tasks.oolong import _to_answer_data

    rows = [
        {
            "id": 42,
            "question": "Which label is most common?",
            "answer": "['incorrect']",
            "context_window_text": "line 1 ...\nline 2 ...",
        }
    ]
    data = _to_answer_data(rows)
    assert data.references["42"] == "incorrect"  # stringified list cleaned
    assert data.corpus_for("42")["context"]["text"].startswith("line 1")
    assert data.gold_by_qid["42"] == ["context"]  # whole context is the gold


def test_multihop_rag_transform():
    from mteb.agentic.tasks.multihop_rag import _to_answer_data

    corpus = [
        {"url": "u1", "title": "T1", "body": "evidence one"},
        {"url": "u2", "title": "T2", "body": "unrelated"},
    ]
    queries = [
        {
            "query": "Who?",
            "answer": "Sam",
            "evidence_list": [{"url": "u1", "title": "T1"}],
        }
    ]
    data = _to_answer_data(corpus, queries)
    assert data.documents["u1"]["text"] == "evidence one"
    assert data.references["0"] == "Sam"
    assert data.gold_by_qid["0"] == ["u1"]


def test_longbench_v2_transform():
    from mteb.agentic.tasks.longbench_v2 import _to_answer_data

    rows = [
        {
            "_id": "q1",
            "question": "Which city?",
            "context": "a long transcript ...",
            "choice_A": "Paris",
            "choice_B": "Berlin",
            "choice_C": "Rome",
            "choice_D": "Madrid",
            "answer": "A",
        }
    ]
    data = _to_answer_data(rows)
    assert data.references["q1"] == "(A) Paris"  # letter + text for the judge
    assert (
        "Which city?" in data.questions["q1"] and "(D) Madrid" in data.questions["q1"]
    )
    assert data.corpus_for("q1")["context"]["text"].startswith("a long transcript")


def test_task_registry_is_official_set():
    # NQ (invented pooled config) was removed; only official-data tasks remain.
    assert set(list_tasks()) == {
        "BrowseCompPlus",
        "HotpotQA",
        "MuSiQue",
        "MultiHopRAG",
        "OOLONG",
        "LongBenchV2",
    }


def test_windowed_full_context_splits_within_document():
    # A single long document must split into multiple overlapping windows, then
    # aggregate; a small document stays one window (no aggregate call).
    model = FakeChatModel([ChatResponse(text=f"part {i}") for i in range(20)])
    system = WindowedFullContextSystem(
        model, window_chars=300_000, overlap_chars=60_000, max_windows=8
    )
    big = InMemoryCorpus({"c": {"text": "x" * 1_000_000}})
    windows, _ = system._windows(big)
    assert len(windows) > 1  # sliding window fires on one big doc
    result = system.answer("q?", big)
    assert result.answer and result.usage.num_llm_calls == len(windows) + 1

    small = InMemoryCorpus({"c": {"text": "x" * 500}})
    assert len(system._windows(small)[0]) == 1


def test_iterative_rag_decompose_loop():
    scripted = [
        ChatResponse(text="capital of France"),  # sub-query 1
        ChatResponse(text="READY"),  # enough evidence
        ChatResponse(text="Paris"),  # final answer
    ]
    agent = IterativeRAG(FakeChatModel(scripted), top_k=1, max_hops=3)
    result = agent.answer("What is the capital of France?", corpus())
    assert result.answer == "Paris"
    assert result.usage.num_tool_calls == 1  # one retrieval before READY
    assert result.cited_doc_ids


def test_retrieval_corpus_reindexes_when_wrapper_clears():
    # Dense/late-interaction wrappers free their index after each search; the
    # corpus must re-index so per-query search keeps working.
    class ClearingSearchModel(FakeSearchModel):
        def __init__(self) -> None:
            self.indexed = False

        def index(self, corpus, **kwargs) -> None:
            super().index(corpus, **kwargs)
            self.indexed = True

        def search(self, queries, *, top_k, **kwargs):
            if not self.indexed:
                raise ValueError("Corpus must be indexed before searching.")
            self.indexed = False  # single-use: cleared after one search
            return super().search(queries, top_k=top_k, **kwargs)

    c = RetrievalCorpus(CORPUS, ClearingSearchModel())
    assert c.search("capital of France", top_k=1)  # first search
    assert c.search("capital of Germany", top_k=1)  # would fail without re-index


def test_harbor_forwards_auth_env(tmp_path, monkeypatch):
    import mteb.agentic.harbor as hb

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = list(cmd)

    monkeypatch.setattr(hb.subprocess, "run", fake_run)
    hb.run_harbor(tmp_path, "claude-code", "m", tmp_path / "jobs")
    joined = " ".join(captured["cmd"])
    assert "OPENAI_API_KEY=sk-test" in joined
    assert "CLAUDE_CODE_OAUTH_TOKEN=oat-test" in joined


def test_harbor_ships_retriever_tool(tmp_path):
    from mteb.agentic.harbor import to_harbor_dataset

    to_harbor_dataset(
        {"q1": "capital of France?"},
        lambda qid: {"a": {"text": "Paris is the capital of France."}},
        tmp_path,
        retriever_tool=True,
    )
    task = tmp_path / "q0"
    assert (task / "environment" / "search.py").exists()
    assert "search.py" in (task / "instruction.md").read_text()


def test_harbor_reads_metrics(tmp_path):
    import json

    from mteb.agentic.harbor import read_harbor_metrics

    trial = tmp_path / "job" / "q0__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_output_tokens": 20,
                    "cost_usd": 0.5,
                },
                "agent_execution": {
                    "started_at": "2026-07-09T15:00:00",
                    "finished_at": "2026-07-09T15:00:30",
                },
            }
        )
    )
    m = read_harbor_metrics(tmp_path)["q0"]
    assert m["prompt_tokens"] == 100 and m["completion_tokens"] == 20
    assert m["cost_usd"] == 0.5 and m["latency_s"] == 30.0


def test_browsecomp_plus_transform():
    # Round-trip the official XOR/base64 scheme so the decrypt path is covered.
    import base64

    from mteb.agentic.tasks.browsecomp_plus import _CANARY, _derive_key, _to_answer_data

    def encrypt(plain: str) -> str:
        data = plain.encode("utf-8")
        key = _derive_key(_CANARY, len(data))
        return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()

    corpus_rows = [{"docid": "d1", "text": "Paris is the capital."}]
    query_rows = [
        {
            "query_id": "q1",
            "query": encrypt("capital of France?"),
            "answer": encrypt("Paris"),
            "gold_docs": [{"docid": encrypt("d1")}],
            "evidence_docs": [{"docid": encrypt("d-missing")}],  # not in corpus
        }
    ]
    data = _to_answer_data(corpus_rows, query_rows)
    assert data.questions["q1"] == "capital of France?"
    assert data.references["q1"] == "Paris"
    assert data.gold_by_qid["q1"] == ["d1"]  # missing evidence doc filtered out


def test_to_scores_dict_bridges_aggregate():
    from mteb.agentic import to_scores_dict

    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    res = evaluate(
        "rag",
        data,
        model=FakeChatModel(),
        judge=ExactMatchJudge(),
        retriever=FakeSearchModel(),
        top_k=1,
    )
    scores = to_scores_dict(res.scores)
    assert scores["accuracy"] == 1.0
    assert scores["mean_recall"] == 1.0
    assert scores["n"] == 1.0 and scores["coverage"] == 1.0


def test_search_agent_forces_final_answer_at_iteration_cap():
    # Every scripted turn calls a tool; at the cap the agent must still answer.
    tool_turn = ChatResponse(
        text="",
        tool_calls=[ToolCall(id="c", name="search", arguments='{"query": "x"}')],
    )
    scripted = [tool_turn, tool_turn, ChatResponse(text="Exact Answer: Paris")]
    agent = SearchAgent(FakeChatModel(scripted), top_k=1, max_iterations=2)
    result = agent.answer("capital?", corpus())
    assert result.answer == "Paris"  # from the forced final call
    assert result.usage.num_llm_calls == 3


def test_evaluate_rejects_unknown_kwargs_but_broadcasts_known():
    data = from_mteb_retrieval(
        CORPUS, {"q1": "capital of France?"}, {"q1": {"d1": 1}}, {"q1": "Paris"}
    )
    # top_k applies to rag and is skipped for closed-book, without error.
    results = evaluate(
        task=data,
        systems=["closed-book", "rag"],
        model=FakeChatModel(),
        judge=ExactMatchJudge(),
        retriever=FakeSearchModel(),
        top_k=1,
    )
    assert set(results) == {"closed-book", "rag"}
    with pytest.raises(TypeError, match="Unknown system kwargs"):
        evaluate(
            "closed-book",
            data,
            model=FakeChatModel(),
            judge=ExactMatchJudge(),
            not_a_kwarg=1,
        )
