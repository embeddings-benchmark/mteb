"""Regression tests for the two `JinaV5OmniWrapper` bugs that diverged our
MTEB scores from the harness frontier dashboard:

1. The wrapper unconditionally injected the retrieval `"Query: " / "Document: "`
   prefix for every task type, including clustering / text-matching /
   classification variants whose LoRA adapters were trained without those
   prefixes. The harness sets `instructions={"query":"","document":""}` for
   non-retrieval variants (see harness `src/core/evaluator.py:578-579`).

2. The nano `ModelMeta` forced `torch_dtype=torch.float32`, while the harness
   runs nano in bf16 (matching small). The forced upcast caused large score
   divergences on OCR / document retrieval tasks for nano only.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_implementations.jina_models import (
    JinaV5OmniWrapper,
    jina_embeddings_v5_omni_nano,
    jina_embeddings_v5_omni_small,
)
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


class _StubFeatures(dict):
    """Dict-like with the `features` attr the wrapper checks via `in`."""


class _StubDataset:
    def __init__(self, features: dict[str, Any]) -> None:
        self.features = _StubFeatures(features)


class _StubDataLoader:
    """Minimal DataLoader stand-in. Yields a single text-only batch."""

    def __init__(self, texts: list[str]) -> None:
        self.dataset = _StubDataset({"text": None})
        self._texts = texts
        self.collate_fn: Any = None

    def __iter__(self):
        yield {"text": list(self._texts)}


class _StubSTModel:
    """Captures the kwargs the wrapper passes to `model.encode`."""

    def __init__(self) -> None:
        self.prompts: dict[str, str] | None = None
        self.captured: list[dict[str, Any]] = []

    def encode(self, inputs, **kwargs):  # noqa: D401, ANN001
        self.captured.append(
            {"inputs": inputs, "task": kwargs.get("task"), "prompt": kwargs.get("prompt")}
        )
        return np.zeros((len(inputs), 4), dtype=np.float32)


# Minimal task-type → variant map mirroring the wrapper's `model_prompts`.
_VARIANT_MAP = {
    "Retrieval": "retrieval",
    "Clustering": "clustering",
    "STS": "text-matching",
    "PairClassification": "text-matching",
    "Classification": "classification",
}


def _make_wrapper() -> tuple[JinaV5OmniWrapper, _StubSTModel]:
    stub = _StubSTModel()
    # Skip the metadata-from-SentenceTransformer path, which reads attributes
    # our stub doesn't have. We only care about the encode-time prompt logic.
    with patch.object(
        ModelMeta, "from_sentence_transformer_model", return_value=ModelMeta.create_empty()
    ):
        wrapper = JinaV5OmniWrapper(model=stub, model_prompts=_VARIANT_MAP)
    return wrapper, stub


def _task(task_type: str) -> TaskMetadata:
    """Build a TaskMetadata that resolves to `task_type` via get_prompt_name."""
    return TaskMetadata(
        name=f"Mock{task_type}",
        description="mock",
        reference=None,
        dataset={"path": "mock", "revision": "mock"},
        type=task_type,
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )


def _encode(wrapper, prompt_type, task_type):
    wrapper.encode(
        _StubDataLoader(["hello"]),
        task_metadata=_task(task_type),
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
    )


def test_retrieval_variant_keeps_query_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.query, "Retrieval")
    assert stub.captured[-1]["task"] == "retrieval"
    assert stub.captured[-1]["prompt"] == "Query: "


def test_retrieval_variant_keeps_document_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.document, "Retrieval")
    assert stub.captured[-1]["task"] == "retrieval"
    assert stub.captured[-1]["prompt"] == "Document: "


def test_clustering_variant_drops_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.document, "Clustering")
    assert stub.captured[-1]["task"] == "clustering"
    assert stub.captured[-1]["prompt"] == ""


def test_text_matching_variant_drops_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.query, "STS")
    assert stub.captured[-1]["task"] == "text-matching"
    assert stub.captured[-1]["prompt"] == ""


def test_classification_variant_drops_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.document, "Classification")
    assert stub.captured[-1]["task"] == "classification"
    assert stub.captured[-1]["prompt"] == ""


def test_pair_classification_variant_drops_prefix():
    wrapper, stub = _make_wrapper()
    _encode(wrapper, PromptType.query, "PairClassification")
    assert stub.captured[-1]["task"] == "text-matching"
    assert stub.captured[-1]["prompt"] == ""


def test_nano_does_not_force_float32_dtype():
    """nano must not pin torch_dtype=float32; harness runs bf16 like small."""
    nano_kwargs = jina_embeddings_v5_omni_nano.loader_kwargs or {}
    small_kwargs = jina_embeddings_v5_omni_small.loader_kwargs or {}
    nano_model_kwargs = nano_kwargs.get("model_kwargs", {}) or {}
    assert nano_model_kwargs.get("torch_dtype") is not torch.float32, (
        "nano forces float32 but harness runs bf16 -- this caused large divergence "
        "on OCR/document tasks (e.g. HatefulMemesT2IRetrieval 0.77 vs 0.06)."
    )
    # And it must match small's dtype handling (small has no override).
    assert "model_kwargs" not in small_kwargs or "torch_dtype" not in (
        small_kwargs.get("model_kwargs") or {}
    )
    assert "model_kwargs" not in nano_kwargs or "torch_dtype" not in (
        nano_kwargs.get("model_kwargs") or {}
    )
