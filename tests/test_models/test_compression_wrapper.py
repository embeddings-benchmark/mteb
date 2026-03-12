import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import mteb
from mteb import AbsTask, EncoderProtocol, TaskMetadata
from mteb.models import CompressionWrapper
from mteb.types import PromptType
from tests.task_grid import MOCK_TASK_TEST_GRID_MONOLINGUAL

task_metadata = TaskMetadata(
    name="DummyTask",
    description="Dummy task metadata",
    dataset={"path": "test", "revision": "test"},
    type="Retrieval",
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
)

# Simulate data
task_texts = DataLoader(
    Dataset.from_dict(
        {
            "text": [
                "What is the effect of vitamin C on common cold?",
                "How does exercise affect cardiovascular health?",
            ]
        }
    )
)


def test_float8_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, torch.float8_e5m2)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert embeddings.dtype == torch.float16


@pytest.mark.parametrize("level, bits", [(torch.int8, 8), (torch.int, 4)])
def test_int_compression(level: str, bits: int):
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, level)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert (
        torch.max(embeddings) <= 2**bits / 2 and torch.min(embeddings) >= -(2**bits) / 2
    )
    assert wrapper.mins is not None and len(wrapper.mins) == len(embeddings[0])
    assert wrapper.maxs is not None and len(wrapper.maxs) == len(embeddings[0])


def test_binary_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, torch.bool)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert torch.max(embeddings) <= 1 and torch.min(embeddings) >= 0


def test_query_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, torch.int8)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
        prompt_type=PromptType.query,
    )
    assert wrapper.quantize_queries
    assert embeddings.dtype == torch.float32


def test_query_compression_multimodal():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, torch.int8)
    metadata = task_metadata.model_copy()
    metadata.category = "t2i"
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=metadata,
        hf_split="test",
        hf_subset="test",
        prompt_type=PromptType.query,
    )
    assert not wrapper.quantize_queries
    assert torch.max(embeddings) <= 2**8 / 2 and torch.min(embeddings) >= -(2**8) / 2


def test_invalid_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, "full")
    with pytest.raises(ValueError):
        wrapper.encode(
            task_texts,
            task_metadata=task_metadata,
            hf_split="test",
            hf_subset="test",
        )


def test_quantize_queries():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, torch.int8)
    query_embeds = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
        prompt_type=PromptType.query,
    )
    doc_embeds = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert wrapper.quantize_queries
    assert wrapper.mins is not None
    wrapper.similarity(doc_embeds, query_embeds)
    assert wrapper.mins is None


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID_MONOLINGUAL)
@pytest.mark.parametrize(
    "model",
    [
        mteb.get_model("mteb/baseline-random-encoder"),
        mteb.get_model(
            "mteb/baseline-random-encoder",
            array_framework="torch",
            dtype=torch.float32,
        ),
        mteb.get_model(
            "mteb/baseline-random-encoder",
            array_framework="torch",
            dtype=torch.float16,
        ),
    ],
)
def test_encoder_dtype_on_task(task: AbsTask, model: EncoderProtocol):
    wrapper = CompressionWrapper(model, torch.float8_e5m2)
    mteb.evaluate(wrapper, task, cache=None)
