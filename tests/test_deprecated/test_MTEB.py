from __future__ import annotations

from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb
from mteb import MTEB, AbsTask
from mteb.models.model_meta import ModelMeta
from tests.test_benchmark.mock_tasks import MockRetrievalTask


@pytest.mark.parametrize(
    "training_datasets",
    [
        {"Touche2020"},  # parent task
        {"Touche2020-NL"},  # child task
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
    expected = sorted(
        [
            "NanoTouche2020Retrieval",
            "Touche2020",
            "Touche2020-Fa",
            "Touche2020-Fa.v2",
            "Touche2020-NL",
            "Touche2020-VN",
            "Touche2020-PL",
            "Touche2020Retrieval.v3",
        ]
    )
    assert sorted(dummy_model_meta.get_training_datasets()) == expected


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

#### --- Create model meta --- ####

def test_create_model_meta_from_sentence_transformers():
    model_name = "sentence-transformers/average_word_embeddings_levy_dependency"
    revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    model = SentenceTransformer(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    assert meta.embed_dim == model.get_sentence_embedding_dimension()
    assert type(meta.framework) is list
    assert meta.framework[0] == "Sentence Transformers"
    assert meta.name == model_name
    assert meta.revision == revision



def test_create_model_meta_from_cross_encoder():
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    revision = "841d331b6f34b15d6ac0ab366ae3a3b36eeac691"
    model = CrossEncoder(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    assert meta.name == model_name
    assert meta.revision == revision

@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_output_folder_model_meta(task: AbsTask, tmp_path: Path):
    mteb = MTEB(tasks=[task])
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    model = CrossEncoder(model_name)
    meta = mteb.create_model_meta(model)
    output_path = mteb.create_output_folder(
        model_meta=meta, output_folder=tmp_path.as_posix()
    )

    output_path = Path(output_path)
    assert output_path.exists()
    assert output_path.is_dir()
    assert output_path.name == model.config._commit_hash
    assert output_path.parent.name == "cross-encoder__ms-marco-TinyBERT-L-2-v2"
    assert output_path.parent.parent == tmp_path
