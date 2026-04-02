import pytest
import torch
from datasets import Dataset
from packaging.version import Version
from torch.utils.data import DataLoader

import mteb
from mteb import AbsTask, EncoderProtocol, TaskMetadata
from mteb.models import CompressionWrapper
from mteb.types import OutputDType
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
        OutputDType.FLOAT8_E4M3FNUZ,
        OutputDType.FLOAT8_E5M2,
        OutputDType.FLOAT8_E5M2FNUZ,
        OutputDType.FLOAT8_E8M0FNU,
        OutputDType.FLOAT8_E4M3FN,
        OutputDType.FLOAT16,
    ],
)
def test_float_compression(level: OutputDType):
    if Version(torch.__version__) < Version("2.10.0"):
        pytest.skip("requires PyTorch")
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
    wrapper = CompressionWrapper(model, OutputDType.BF16)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    # Cast to bf16, then back to float32 using PyTorch as numpy doesn't support bf16
    assert embeddings.dtype == torch.float32


@pytest.mark.parametrize("level, bits", [(OutputDType.INT8, 8), (OutputDType.INT4, 4)])
def test_int_compression(level: OutputDType, bits: int):
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


@pytest.mark.parametrize(
    "level, bits", [(OutputDType.UINT8, 8), (OutputDType.UINT4, 4)]
)
def test_uint_compression(level: OutputDType, bits: int):
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, level)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert torch.max(embeddings) <= 2**bits and torch.min(embeddings) >= 0


def test_binary_compression():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = CompressionWrapper(model, OutputDType.BINARY)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert torch.max(embeddings) <= 1 and torch.min(embeddings) >= 0


@pytest.mark.parametrize("margins", [(-0.1, 0.7), (0.1, 1.3), (0.2, 0.1)])
def test_invalid_clipping_margin(margins: tuple[float, float]):
    model = mteb.get_model("mteb/baseline-random-encoder")
    with pytest.raises(ValueError):
        CompressionWrapper(model, OutputDType.INT8, margins)


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
    wrapper = CompressionWrapper(model, OutputDType.BINARY)
    mteb.evaluate(wrapper, task, cache=None)
