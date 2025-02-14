from __future__ import annotations

import pytest

import mteb


@pytest.mark.parametrize(
    ("model_name", "expected_memory"),
    [
        ("intfloat/e5-mistral-7b-instruct", 13563),  # multiple safetensors
        ("infgrad/jasper_en_vision_language_v1", 3802),  # bf16
        ("intfloat/multilingual-e5-small", 449),  # safetensors
        ("BAAI/bge-m3", 2167),  # pytorch_model.bin
    ],
)
def test_model_memory_usage(model_name: str, expected_memory: int | None):
    meta = mteb.get_model_meta(model_name)
    assert meta.memory_usage_mb is not None
    used_memory = round(meta.memory_usage_mb)
    assert used_memory == expected_memory


def test_model_memory_usage_api_model():
    meta = mteb.get_model_meta("openai/text-embedding-3-large")
    assert meta.memory_usage_mb is None
