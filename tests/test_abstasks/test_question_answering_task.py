"""End-to-end tests for AbsTaskQuestionAnswering through mteb.evaluate."""

from __future__ import annotations

from datasets import Dataset

import mteb
from mteb.abstasks.question_answering import AbsTaskQuestionAnswering
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.answer_systems import (
    AnswerProtocol,
    ClosedBookAnswerer,
    ExactMatchJudge,
    LLMJudge,
    OracleAnswerer,
    RAGAnswerer,
)
from mteb.models.chat_models import ChatResponse

CORPUS = Dataset.from_list(
    [
        {"id": "d1", "title": "", "text": "Paris is the capital of France."},
        {"id": "d2", "title": "", "text": "Berlin is the capital of Germany."},
    ]
)
QUERIES = Dataset.from_list([{"id": "q1", "text": "What is the capital of France?"}])
ANSWERS = {"q1": "Paris"}
RELEVANT = {"q1": {"d1": 1}}


class FakeChatModel:
    def __init__(self, scripted: list[str]) -> None:
        self._scripted = list(scripted)

    def generate(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(text=self._scripted.pop(0), cost_usd=0.001)


class FakeSearchModel:
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
            out[row["id"]] = dict(sorted(scored.items(), key=lambda kv: -kv[1])[:top_k])
        return out


class TinyAnswerTask(AbsTaskQuestionAnswering):
    metadata = TaskMetadata(
        dataset={"path": "mteb/test", "revision": "main"},
        name="TinyAnswerTask",
        description="Fixture task for answer-mode tests.",
        type="QuestionAnswering",
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )

    def load_data(self, **kwargs) -> None:
        self.dataset = {
            "default": {
                "test": {"corpus": CORPUS, "queries": QUERIES, "answers": ANSWERS}
            }
        }
        self.data_loaded = True


def test_answer_task_through_mteb_evaluate():
    rag = RAGAnswerer(FakeChatModel(["Paris"]), FakeSearchModel(), top_k=1)
    assert isinstance(rag, AnswerProtocol)
    res = mteb.evaluate(rag, TinyAnswerTask(), cache=None)
    scores = res[0].scores["test"][0]
    assert scores["accuracy"] == 1.0
    assert scores["main_score"] == 1.0
    assert scores["cost_usd"] == 0.001


def test_floor_and_ceiling_baselines():
    task = TinyAnswerTask()
    closed = ClosedBookAnswerer(FakeChatModel(["Rome"]))  # wrong from memory
    oracle = OracleAnswerer(FakeChatModel(["Paris"]), RELEVANT)
    low = mteb.evaluate(closed, task, cache=None)[0].scores["test"][0]
    high = mteb.evaluate(oracle, task, cache=None)[0].scores["test"][0]
    assert low["accuracy"] == 0.0 and high["accuracy"] == 1.0


def test_judges():
    assert ExactMatchJudge().score("q", "  PARIS. ", "Paris") == 1.0
    yes = LLMJudge(FakeChatModel(["correct: yes"]))
    assert yes.score("q", "The capital is Paris", "Paris") == 1.0
    no = LLMJudge(FakeChatModel(["correct: no"]))
    assert no.score("q", "Berlin", "Paris") == 0.0
