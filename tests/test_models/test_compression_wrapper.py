import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import mteb
from mteb import AbsTask, EncoderProtocol, TaskMetadata
from mteb.models import CompressionWrapper
from mteb.types import QuantizationLevel
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


@pytest.mark.parametrize(
    "level",
    [
        QuantizationLevel.FLOAT8_E4M3FNUZ,
        QuantizationLevel.FLOAT8_E5M2,
        QuantizationLevel.FLOAT8_E5M2FNUZ,
        QuantizationLevel.FLOAT8_E8M0FNU,
        QuantizationLevel.FLOAT8_E4M3FN,
        QuantizationLevel.FLOAT16,
    ],
)
def test_float_compression(level: QuantizationLevel):
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, level)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert embeddings.dtype == torch.float16


def test_bf16_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, QuantizationLevel.BF16)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert embeddings.dtype == torch.float32


@pytest.mark.parametrize(
    "level, bits", [(QuantizationLevel.INT8, 8), (QuantizationLevel.INT4, 4)]
)
def test_int_compression(level: QuantizationLevel, bits: int):
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


def test_binary_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, QuantizationLevel.BINARY)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert torch.max(embeddings) <= 1 and torch.min(embeddings) >= 0


def test_invalid_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, QuantizationLevel.UINT8)
    with pytest.raises(ValueError) as e:
        wrapper.encode(
            task_texts,
            task_metadata=task_metadata,
            hf_split="test",
            hf_subset="test",
        )
    print(e)


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
    wrapper = CompressionWrapper(model, QuantizationLevel.FLOAT8_E4M3FN)
    mteb.evaluate(wrapper, task, cache=None)
