from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scripts.generate_metadata import get_base_model
from scripts.generate_metadata import main as generate_metadata_main


def test_create_model_meta_embedding_models_from_hf(tmp_path: Path):
    models = ["intfloat/multilingual-e5-large", "intfloat/multilingual-e5-small"]
    tmp_path = tmp_path / "new_models.py"
    generate_metadata_main(tmp_path, models)

    assert tmp_path.exists()
    assert tmp_path.read_text().startswith("from mteb.model_meta import ModelMeta")

    spec = importlib.util.spec_from_file_location("new_models", tmp_path)
    new_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_models)

    assert hasattr(new_models, "intfloat__multilingual_e5_large")
    assert hasattr(new_models, "intfloat__multilingual_e5_small")

    assert (
        new_models.intfloat__multilingual_e5_large.name
        == "intfloat/multilingual-e5-large"
    )
    assert (
        new_models.intfloat__multilingual_e5_small.name
        == "intfloat/multilingual-e5-small"
    )


def test_get_base_model_name_is_the_same():
    model_name = "jinaai/jina-embeddings-v3"
    model = get_base_model(model_name)
    assert model is None


@pytest.mark.skip(reason="No support for cross-encoder models")
def test_create_model_meta_cross_encoder_models_from_hf(tmp_path: Path):
    models = ["intfloat/multilingual-e5-cross-encoder"]
    tmp_path = tmp_path / "new_models.py"
    generate_metadata_main(tmp_path, models)
    assert True
