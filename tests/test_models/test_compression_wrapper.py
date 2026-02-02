import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb import TaskMetadata
from mteb.models import CompressionWrapper
from mteb.models.model_implementations.random_baseline import (
    RandomEncoderBaseline,
    random_encoder_baseline,
)
from mteb.types import PromptType

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
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    model.mteb_model_meta = random_encoder_baseline
    model.mteb_model_meta.name = "dummy"
    wrapper = CompressionWrapper(model, "float8")
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert model.mteb_model_meta.name == "dummy (output_dtype=float8)"
    assert embeddings.dtype == np.float16


@pytest.mark.parametrize("level, bits", [("int8", 8), ("int4", 4)])
def test_int_compression(level: str, bits: int):
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    wrapper = CompressionWrapper(model, level)
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert np.max(embeddings) <= 2**bits / 2 and np.min(embeddings) >= -(2**bits) / 2
    assert wrapper.mins is not None and len(wrapper.mins) == len(embeddings[0])
    assert wrapper.maxs is not None and len(wrapper.maxs) == len(embeddings[0])


def test_binary_compression():
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    wrapper = CompressionWrapper(model, "binary")
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert np.max(embeddings) <= 1 and np.min(embeddings) >= 0


def test_query_compression():
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    wrapper = CompressionWrapper(model, "int8")
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
        prompt_type=PromptType.query,
    )
    assert np.array_equal(embeddings, wrapper.query_embeds)


def test_query_compression_multimodal():
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    wrapper = CompressionWrapper(model, "int8")
    metadata = task_metadata.model_copy()
    metadata.category = "t2i"
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=metadata,
        hf_split="test",
        hf_subset="test",
        prompt_type=PromptType.query,
    )
    assert wrapper.query_embeds is None
    assert np.max(embeddings) <= 2**8 / 2 and np.min(embeddings) >= -(2**8) / 2


def test_invalid_compression():
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    wrapper = CompressionWrapper(model, "full")
    with pytest.raises(ValueError) as e:
        wrapper.encode(
            task_texts,
            task_metadata=task_metadata,
            hf_split="test",
            hf_subset="test",
        )
    assert str(e.value) == "Quantization method full is not supported!"


def test_quantize_queries():
    model = RandomEncoderBaseline(model_name="dummy", revision=None)
    model.mteb_model_meta = random_encoder_baseline
    wrapper = CompressionWrapper(model, "int8")
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
    assert np.array_equal(query_embeds, wrapper.query_embeds)
    assert wrapper.mins is not None
    wrapper.similarity(doc_embeds, query_embeds)
    assert wrapper.mins is None
