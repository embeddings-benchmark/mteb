"""Answer systems: produce an answer to a query over a fixed corpus.

Systems follow the retrieval model lifecycle: index(corpus) once, then
answer(query_id, question) per query. Correctness is scored by a Judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.chat_models import ChatModelProtocol
    from mteb.models.models_protocols import SearchProtocol
    from mteb.types import CorpusDatasetType, EncodeKwargs

_ANSWER = "Answer the question concisely.\n\nQuestion: {q}"
_CONTEXT = (
    "Answer the question concisely using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {q}"
)


@dataclass
class AnswerResult:
    """One produced answer with cost accounting."""

    text: str
    cost_usd: float | None = None


@runtime_checkable
class AnswerProtocol(Protocol):
    """Interface for answer systems evaluated by AbsTaskQuestionAnswering."""

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        """Prepare the corpus before answering."""
        ...

    def answer(self, query_id: str, question: str) -> AnswerResult:
        """Answer one question."""
        ...


def _answer_meta(kind: str, model: ChatModelProtocol) -> ModelMeta:
    llm_name = (getattr(model, "name", None) or "llm").rsplit("/", 1)[-1]
    return ModelMeta.create_empty(
        overwrites={"name": f"mteb/answer-{kind}-{llm_name}", "model_type": ["hybrid"]}
    )


class ClosedBookAnswerer:
    """Floor baseline: answers from the model alone, ignoring the corpus."""

    def __init__(self, model: ChatModelProtocol) -> None:
        self.model = model
        self.mteb_model_meta = _answer_meta("closed-book", model)

    def index(self, corpus: CorpusDatasetType, **kwargs: object) -> None:
        """No corpus access."""

    def answer(self, query_id: str, question: str) -> AnswerResult:
        """Answer from parametric memory."""
        out = self.model.generate(
            [{"role": "user", "content": _ANSWER.format(q=question)}]
        )
        return AnswerResult(text=out.text, cost_usd=out.cost_usd)


class _ContextAnswerer:
    """Base for systems that answer from selected document text."""

    def __init__(self, model: ChatModelProtocol, kind: str, snippet_chars: int) -> None:
        self.model = model
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _answer_meta(kind, model)
        self._text: dict[str, str] = {}

    def _answer_from(self, question: str, doc_ids: list[str]) -> AnswerResult:
        context = "\n\n".join(
            self._text.get(d, "")[: self.snippet_chars] for d in doc_ids
        )
        out = self.model.generate(
            [{"role": "user", "content": _CONTEXT.format(context=context, q=question)}]
        )
        return AnswerResult(text=out.text, cost_usd=out.cost_usd)


class RAGAnswerer(_ContextAnswerer):
    """Retrieve top_k with the wrapped retriever, then answer from the context."""

    def __init__(
        self,
        model: ChatModelProtocol,
        retriever: SearchProtocol,
        *,
        top_k: int = 5,
        snippet_chars: int = 2000,
    ) -> None:
        super().__init__(model, "rag", snippet_chars)
        self.retriever = retriever
        self.top_k = top_k
        self._search_kwargs: dict[str, object] = {}

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        """Index the retriever and cache document text."""
        self._text = {row["id"]: row.get("text", "") for row in corpus}
        self._search_kwargs = {
            "task_metadata": task_metadata,
            "hf_split": hf_split,
            "hf_subset": hf_subset,
            "encode_kwargs": encode_kwargs,
            "num_proc": num_proc,
        }
        self.retriever.index(corpus, **self._search_kwargs)  # type: ignore[arg-type]

    def answer(self, query_id: str, question: str) -> AnswerResult:
        """Retrieve once, then answer from the retrieved context."""
        from datasets import Dataset

        ranking = self.retriever.search(
            Dataset.from_list([{"id": query_id, "text": question}]),
            top_k=self.top_k,
            **self._search_kwargs,  # type: ignore[arg-type]
        )[query_id]
        doc_ids = sorted(ranking, key=lambda d: -ranking[d])
        return self._answer_from(question, doc_ids)


class OracleAnswerer(_ContextAnswerer):
    """Ceiling baseline: answers from the gold documents for each query."""

    def __init__(
        self,
        model: ChatModelProtocol,
        relevant_docs: dict[str, dict[str, int]],
        *,
        snippet_chars: int = 2000,
    ) -> None:
        super().__init__(model, "oracle", snippet_chars)
        self.relevant_docs = relevant_docs

    def index(self, corpus: CorpusDatasetType, **kwargs: object) -> None:
        """Cache document text for gold-context answering."""
        self._text = {row["id"]: row.get("text", "") for row in corpus}

    def answer(self, query_id: str, question: str) -> AnswerResult:
        """Answer from the labeled gold documents."""
        gold = [d for d, s in self.relevant_docs.get(query_id, {}).items() if s > 0]
        return self._answer_from(question, gold)


@runtime_checkable
class JudgeProtocol(Protocol):
    """Scores answer correctness in [0, 1]."""

    def score(self, question: str, prediction: str, reference: str) -> float:
        """Grade a predicted answer against the reference."""
        ...


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return " ".join(text.split())


class ExactMatchJudge:
    """Normalized exact match, for verifiable short answers."""

    def score(self, question: str, prediction: str, reference: str) -> float:  # noqa: PLR6301
        """Return 1.0 on a normalized match, else 0.0."""
        return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


_JUDGE = (
    "Grade whether the predicted answer matches the reference answer. Extract "
    "the final answer from the prediction and judge only whether it matches "
    "the reference, ignoring phrasing and extra words. Reply with one line, "
    "exactly 'correct: yes' or 'correct: no'.\n\n"
    "Question: {q}\nReference answer: {ref}\nPrediction: {pred}"
)
_VERDICT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)


class LLMJudge:
    """LLM grading of open-ended answers (BrowseComp grading scheme)."""

    def __init__(self, model: ChatModelProtocol) -> None:
        self.model = model

    def score(self, question: str, prediction: str, reference: str) -> float:
        """Return 1.0 if the judge deems the prediction correct."""
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _JUDGE.format(
                        q=question, ref=reference, pred=prediction
                    ),
                }
            ]
        )
        match = _VERDICT_RE.search(out.text)
        return 1.0 if match and match.group(1).lower() == "yes" else 0.0
