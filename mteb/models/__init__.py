from __future__ import annotations

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.model_meta import ModelMeta
from mteb.models import e5_models, sentence_transformers_models


def get_model(model_name: str) -> Encoder | EncoderWithQueryCorpusEncode:
    """A function to fetch a model object by name.

    Args:
        model_name: Name of the model to fetch

    Returns:
        A model object
    """
    return models[model_name].load_model()


def get_model_meta(model_name: str) -> ModelMeta:
    """A function to fetch a model metadata object by name.

    Args:
        model_name: Name of the model to fetch

    Returns:
        A model metadata object
    """
    return models[model_name]


model_modules = [e5_models, sentence_transformers_models]
models = {}


for module in model_modules:
    for mdl in module.__dict__.values():
        if isinstance(mdl, ModelMeta):
            models[mdl.name] = mdl
