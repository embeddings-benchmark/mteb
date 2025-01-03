from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from mteb import get_model, get_model_meta
from mteb.models.overview import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)


CACHE_FOLDER = Path(__file__).parent / ".cache"


def teardown_function():
    """Remove cache folder and its contents"""
    for item in CACHE_FOLDER.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_get_all_models_below_n_param_threshold(model_name: str):
    """Test that we can get all models with a number of parameters below a threshold."""
    model_meta = get_model_meta(model_name=model_name)
    assert model_meta is not None
    if model_meta.n_parameters is not None and model_meta.n_parameters < 2e9:
        m = get_model(model_name, cache_folder=CACHE_FOLDER)
        assert m is not None, f"Failed to load model {model_name}"
