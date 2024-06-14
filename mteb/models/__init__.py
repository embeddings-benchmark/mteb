from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.model_meta import ModelMeta
from mteb.models import (
    e5_models,
    openai_models,
    sentence_transformers_models,
    voyage_models,
)

logger = logging.getLogger(__name__)


def get_model(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> Encoder | EncoderWithQueryCorpusEncode:
    """A function to fetch a model object by name.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        **kwargs: Additional keyword arguments to pass to the model loader

    Returns:
        A model object
    """
    meta = get_model_meta(model_name, revision)
    model = meta.load_model(**kwargs)

    # If revision not available in the modelmeta, try to extract it from sentence-transformers
    if meta.revision is None and isinstance(model, SentenceTransformer):
        _meta = model_meta_from_sentence_transformers(model)
        meta.revision = _meta.revision if _meta.revision else meta.revision

    model.mteb_model_meta = meta  # type: ignore
    return model


def get_model_meta(model_name: str, revision: str | None = None) -> ModelMeta:
    """A function to fetch a model metadata object by name.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch

    Returns:
        A model metadata object
    """
    if model_name in models:
        if not models[model_name].revision == revision:
            raise ValueError(f"Model revision {revision} not found for model {model_name}")
        return models[model_name]
    else:  # assume it is a sentence-transformers model
        logger.info(
            "Model not found in model registry, assuming it is a sentence-transformers model."
        )
        logger.info(
            f"Attempting to extract metadata by loading the model ({model_name}) using sentence-transformers."
        )
        model = SentenceTransformer(model_name, revision=revision)
        meta = model_meta_from_sentence_transformers(model)

        meta.revision = revision
        meta.name = model_name
    return meta


def model_meta_from_sentence_transformers(model: SentenceTransformer) -> ModelMeta:
    try:
        name = (
            model.model_card_data.model_name
            if model.model_card_data.model_name
            else model.model_card_data.base_model
        )
        languages = (
            [model.model_card_data.language]
            if isinstance(model.model_card_data.language, str)
            else model.model_card_data.language
        )
        meta = ModelMeta(
            name=name,
            revision=model.model_card_data.base_model_revision,
            release_date=None,
            languages=languages,
            framework=["Sentence Transformers"],
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to extract metadata from model: {e}. Upgrading to sentence-transformers v3.0.0 or above is recommended."
        )
        meta = ModelMeta(
            name=None,
            revision=None,
            languages=None,
            release_date=None,
        )
    return meta


model_modules = [e5_models, sentence_transformers_models, openai_models, voyage_models]
models = {}


for module in model_modules:
    for mdl in vars(module).values():
        if isinstance(mdl, ModelMeta):
            models[mdl.name] = mdl
