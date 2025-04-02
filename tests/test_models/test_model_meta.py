from __future__ import annotations

import pytest

import mteb
from mteb import ModelMeta


@pytest.mark.parametrize(
    ("model_name", "expected_memory"),
    [
        ("intfloat/e5-mistral-7b-instruct", 13563),  # multiple safetensors
        ("NovaSearch/jasper_en_vision_language_v1", 3802),  # bf16
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


@pytest.mark.parametrize(
    "training_datasets",
    [
        {"Touche2020": []},  # parent task
        {"Touche2020-NL": []},  # child task
    ],
)
def test_model_similar_tasks(training_datasets):
    dummy_model_meta = ModelMeta(
        name="test/test_model",
        revision="test",
        release_date=None,
        languages=None,
        loader=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        open_weights=None,
        public_training_code=None,
        public_training_data=None,
        framework=[],
        reference=None,
        similarity_fn_name=None,
        use_instructions=None,
        training_datasets=training_datasets,
        adapted_from=None,
        superseded_by=None,
    )
    expected = [
        "NanoTouche2020Retrieval",
        "Touche2020",
        "Touche2020-Fa",
        "Touche2020-NL",
        "Touche2020Retrieval.v3",
    ]
    assert sorted(dummy_model_meta.get_training_datasets().keys()) == expected


def test_model_name_without_prefix():
    with pytest.raises(ValueError):
        ModelMeta(
            name="test_model",
            revision="test",
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            reference=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
        )


def test_model_training_dataset_adapted():
    model_meta = mteb.get_model_meta("deepvk/USER-bge-m3")
    assert model_meta.adapted_from == "BAAI/bge-m3"
    # MIRACLRetrieval not in training_datasets of deepvk/USER-bge-m3, but in
    # training_datasets of BAAI/bge-m3
    assert "MIRACLRetrieval" in model_meta.get_training_datasets()
