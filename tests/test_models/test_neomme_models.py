import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask, MockRetrievalTask
from mteb.models.model_implementations.neomme_models import (
    NeoMMERetrieverWrapper,
    neomme_260m_retriever,
)
from mteb.models.model_meta import ScoringFunction
from mteb.types import Array, PromptType


def _encode(
    wrapper: NeoMMERetrieverWrapper, batch: dict, prompt_type: PromptType | None = None
) -> Array:
    return wrapper.encode(
        [batch],
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
    )


class FakeProcessor:
    image_token = "<img>"  # noqa: S105

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = lambda texts, **kwargs: {
            "offset_mapping": [
                [match.span() for match in re.finditer(r"\S+", text)] for text in texts
            ]
        }

    def apply_chat_template(self, messages: list, **kwargs: Any) -> BatchFeature:
        # Template overhead: a task marker, plus 10 expansion tokens for queries.
        overhead = 11 if kwargs["task"] == "query" else 1
        lengths = [
            len(message[0]["content"].split()) + overhead
            if isinstance(message[0]["content"], str)
            else 4
            for message in messages
        ]
        if "processor_kwargs" not in kwargs:
            return BatchFeature(
                {"input_ids": [list(range(length)) for length in lengths]}
            )
        self.calls.append({"messages": messages, **kwargs})
        ids = torch.arange(max(lengths)).repeat(len(lengths), 1)
        mask = ids < torch.tensor(lengths)[:, None]
        return BatchFeature({"input_ids": ids, "attention_mask": mask.long()})


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=1024)
        self.calls: list[dict[str, Any]] = []

    def to(self, device: str) -> Self:
        return self

    def eval(self) -> Self:
        return self

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> SimpleNamespace:
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled()
        vectors = torch.nn.functional.one_hot(input_ids, num_classes=128).float()
        vectors *= attention_mask[:, :, None]
        dense = torch.nn.functional.pad(
            vectors.sum(dim=1), (0, self.config.hidden_size - 128)
        )
        return SimpleNamespace(
            embeddings=vectors if kwargs["output_multivector"] else None,
            dense_embeddings=torch.nn.functional.normalize(dense, dim=-1)
            if kwargs["output_dense"]
            else None,
        )


@pytest.fixture
def loaders(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    transformers = pytest.importorskip("transformers", minversion="5.17.0")
    model_loader = Mock(side_effect=lambda *args, **kwargs: FakeModel())
    processor_loader = Mock(side_effect=lambda *args, **kwargs: FakeProcessor())
    monkeypatch.setattr(
        transformers.NeoMMEForRetrieval, "from_pretrained", model_loader
    )
    monkeypatch.setattr(
        transformers.NeoMMEProcessor, "from_pretrained", processor_loader
    )
    return model_loader, processor_loader


@pytest.mark.parametrize(
    ("vector_type", "dimension", "model_type", "scoring"),
    [
        ("single_vector", 1024, "dense", ScoringFunction.COSINE),
        ("multi_vector", 128, "late-interaction", ScoringFunction.MAX_SIM),
    ],
)
def test_load_metadata_and_revision(
    loaders: tuple[Mock, Mock],
    vector_type: str,
    dimension: int,
    model_type: str,
    scoring: ScoringFunction,
) -> None:
    model = mteb.get_model(
        neomme_260m_retriever.name, vector_type=vector_type, device="cpu"
    )
    meta = model.mteb_model_meta
    assert meta.embed_dim == dimension
    assert meta.model_type == [model_type]
    assert meta.similarity_fn_name == scoring
    assert meta.revision == neomme_260m_retriever.revision
    for loader in loaders:
        assert loader.call_args.kwargs["revision"] == meta.revision
        assert loader.call_args.kwargs["trust_remote_code"] is False
    assert loaders[0].call_args.kwargs["dtype"] == torch.float32
    assert meta.load_model().mteb_model_meta.embed_dim == dimension
    reloaded = neomme_260m_retriever.load_model(
        vector_type=vector_type, embed_dim=dimension
    )
    assert reloaded.mteb_model_meta.embed_dim == dimension


@pytest.mark.parametrize("vector_type", ["single_vector", "multi_vector"])
def test_head_outputs_and_query_template(
    loaders: tuple[Mock, Mock], vector_type: str
) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model", vector_type=vector_type)
    embeddings = _encode(
        wrapper, {"text": ["short", "a longer query"]}, PromptType.query
    )
    assert wrapper.model.calls[0] == {
        "output_dense": vector_type == "single_vector",
        "output_multivector": vector_type == "multi_vector",
    }
    if vector_type == "single_vector":
        assert embeddings.shape == (2, 1024)
    else:
        assert [tuple(row.shape) for row in embeddings] == [(12, 128), (14, 128)]


def test_image_uses_document_template(loaders: tuple[Mock, Mock]) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model")
    result = _encode(wrapper, {"image": [Image.new("RGB", (32, 32))]}, PromptType.query)
    call = wrapper.processor.calls[0]
    assert call["task"] == "document"
    assert call["processor_kwargs"] == {"padding": "longest"}
    assert result[0].shape == (4, 128)


def test_text_image_token_is_escaped(loaders: tuple[Mock, Mock]) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model")
    _encode(wrapper, {"text": ["see <img> here"]}, PromptType.document)
    assert wrapper.processor.calls[0]["messages"][0][0]["content"] == (
        "see &lt;img&gt; here"
    )


@pytest.mark.parametrize("vector_type", ["single_vector", "multi_vector"])
def test_grouped_batches_match_one_per_batch(
    loaders: tuple[Mock, Mock], vector_type: str
) -> None:
    texts = ["a b c d", "e", "f g", "h i j"]
    batch = {"text": texts}
    one_per_batch = NeoMMERetrieverWrapper(
        "test/model", vector_type=vector_type, max_batch_tokens=1
    )
    grouped = NeoMMERetrieverWrapper("test/model", vector_type=vector_type)
    expected = _encode(one_per_batch, batch)
    actual = _encode(grouped, batch)
    assert len(grouped.processor.calls) == 1
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right)


@pytest.mark.parametrize("prompt_type", [PromptType.document, PromptType.query])
@pytest.mark.parametrize(
    ("text", "expected_content"),
    [
        ("one two three four five", "one two three four "),
        ("one two three four", "one two three four"),
    ],
)
def test_text_is_cut_at_max_text_tokens(
    loaders: tuple[Mock, Mock],
    prompt_type: PromptType,
    text: str,
    expected_content: str,
) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model", max_text_tokens=4)
    _encode(wrapper, {"text": [text]}, prompt_type)
    assert wrapper.processor.calls[0]["messages"][0][0]["content"] == expected_content


def test_token_budget_counts_template_tokens(loaders: tuple[Mock, Mock]) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model", max_batch_tokens=2)
    _encode(wrapper, {"text": ["", "", ""]}, PromptType.document)
    assert len(wrapper.processor.calls) == 2


@pytest.mark.parametrize("vector_type", ["single_vector", "multi_vector"])
def test_encode_across_loader_batches(
    loaders: tuple[Mock, Mock], vector_type: str
) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model", vector_type=vector_type)
    image = Image.new("RGB", (32, 32))
    result = wrapper.encode(
        [
            {"text": ["a b", "c"]},
            {"image": [image]},
            {"text": ["", "d e f", "g"], "image": [image, image, None]},
        ],
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.document,
    )
    if vector_type == "single_vector":
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
        return
    # Documents add 1 marker token, each fake image is 4 tokens, "" + image is image only.
    assert [len(row) for row in result] == [3, 2, 4, 4, 8, 2]


def test_maxsim_is_averaged_over_query_tokens(loaders: tuple[Mock, Mock]) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model")
    generator = torch.Generator().manual_seed(0)
    queries = [torch.randn(length, 8, generator=generator) for length in (1, 5, 3)]
    documents = [torch.randn(length, 8, generator=generator) for length in (4, 1, 7, 2)]
    expected = torch.tensor(
        [
            [(query @ document.T).amax(dim=1).mean().item() for document in documents]
            for query in queries
        ]
    )
    torch.testing.assert_close(wrapper.similarity(queries, documents).cpu(), expected)


@pytest.mark.parametrize("vector_type", ["single_vector", "multi_vector"])
def test_pairwise_agrees_with_matrix(
    loaders: tuple[Mock, Mock], vector_type: str
) -> None:
    wrapper = NeoMMERetrieverWrapper("test/model", vector_type=vector_type)
    queries = _encode(wrapper, {"text": ["one", "two three"]})
    documents = _encode(
        wrapper, {"text": ["four five six", "seven"]}, PromptType.document
    )
    matrix = wrapper.similarity(queries, documents)
    pairwise = wrapper.similarity_pairwise(queries, documents)
    torch.testing.assert_close(pairwise, matrix.diagonal())


def test_invalid_options_fail_before_loading(loaders: tuple[Mock, Mock]) -> None:
    with pytest.raises(ValueError, match="vector_type"):
        NeoMMERetrieverWrapper("test/model", vector_type="invalid")
    with pytest.raises(ValueError, match="vector_type"):
        neomme_260m_retriever.load_model(vector_type="invalid")
    with pytest.raises(ValueError, match="embed_dim"):
        NeoMMERetrieverWrapper("test/model", embed_dim=1024)
    loaders[0].assert_not_called()


def test_heads_have_separate_result_paths(
    loaders: tuple[Mock, Mock], tmp_path: Path
) -> None:
    cache = mteb.ResultCache(tmp_path)

    def result_path(**kwargs: str) -> Path:
        meta = neomme_260m_retriever.load_model(**kwargs).mteb_model_meta
        return cache.get_task_result_path("task", meta)

    plain = cache.get_task_result_path("task", neomme_260m_retriever)
    assert result_path() == result_path(vector_type="multi_vector") == plain
    assert result_path(vector_type="single_vector") != plain


@pytest.mark.parametrize("vector_type", ["single_vector", "multi_vector"])
def test_reranking_runs_end_to_end(
    loaders: tuple[Mock, Mock], tmp_path: Path, vector_type: str
) -> None:
    model = neomme_260m_retriever.load_model(device="cpu", vector_type=vector_type)
    results = mteb.evaluate(
        model, MockRerankingTask(), cache=mteb.ResultCache(tmp_path)
    )
    assert 0.0 <= results.task_results[0].get_score() <= 1.0
