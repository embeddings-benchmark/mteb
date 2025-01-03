from __future__ import annotations

import logging

import pytest

from mteb import get_model
from mteb.models.overview import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_get_all_models_below_n_param_threshold(model_name: str):
    """Test that we can get all models with a number of parameters below a threshold."""
    model_meta = MODEL_REGISTRY.get_model_meta(model_name=model_name)
    assert model_meta is not None
    if model_meta.n_parameters is not None and model_meta.n_parameters < 2e9:
        m = get_model(model_name)
        assert m is not None
