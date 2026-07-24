"""Tests for the answer-mode retrieval core (mteb.agentic)."""

from __future__ import annotations

from mteb.agentic import (
    AnswerEvaluator,
    ChatResponse,
    ClosedBookSystem,
    ExactMatchJudge,
    InMemoryCorpus,
    LLMJudge,
    OracleContextSystem,
    from_mteb_retrieval,
)


class _FakeChat:
    """Deterministic ChatModel for tests, no API key needed."""

    name = "fake"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, messages, **kwargs):
        return ChatResponse(
            text=self._reply, prompt_tokens=5, completion_tokens=1, cost_usd=0.001
        )


def _data():
    return from_mteb_retrieval(
        {"d1": {"title": "France", "text": "The capital of France is Paris."}},
        {"q1": "What is the capital of France?"},
        {"q1": {"d1": 1}},
        {"q1": "Paris"},
    )


def _evaluator():
    data = _data()
    return data, AnswerEvaluator(
        data.questions,
        data.references,
        InMemoryCorpus(data.documents),
        ExactMatchJudge(),
    )


def test_exact_match_judge_normalizes():
    judge = ExactMatchJudge()
    assert judge.score("q", "Paris.", "paris") == 1.0
    assert judge.score("q", "Berlin", "Paris") == 0.0


def test_from_mteb_retrieval_builds_gold():
    data = _data()
    assert data.references["q1"] == "Paris"
    assert data.gold_by_qid["q1"] == ["d1"]
    assert data.gold_by_question["What is the capital of France?"] == ["d1"]


def test_oracle_ceiling_scores_perfectly():
    data, evaluator = _evaluator()
    result = evaluator(
        OracleContextSystem(_FakeChat("Paris"), gold=data.gold_by_question)
    )
    assert result.scores.accuracy == 1.0
    assert result.scores.n == 1
    assert result.scores.mean_cost_usd == 0.001
    assert result.per_question[0]["cited_doc_ids"] == ["d1"]


def test_closed_book_floor_runs():
    _, evaluator = _evaluator()
    result = evaluator(ClosedBookSystem(_FakeChat("Berlin")))
    assert result.scores.accuracy == 0.0
    assert result.scores.mean_llm_calls == 1.0


def test_latency_is_captured():
    _, evaluator = _evaluator()
    result = evaluator(ClosedBookSystem(_FakeChat("Paris")))
    assert result.per_question[0]["latency_s"] is not None


def test_llm_judge_robust_to_reasoning():
    yes = LLMJudge(
        _FakeChat("<think>it matches</think> The prediction is correct. YES")
    )
    no = LLMJudge(_FakeChat("Let me think... this does not match. NO"))
    assert yes.score("q", "QX-4417", "QX-4417") == 1.0
    assert no.score("q", "ZZ-9", "QX-4417") == 0.0
