from __future__ import annotations

import difflib
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from huggingface_hub import ModelCard
from huggingface_hub.errors import RepositoryNotFoundError

from mteb.abstasks import AbsTask
from mteb.models import (
    CrossEncoderWrapper,
    ModelMeta,
    MTEBModels,
    sentence_transformers_loader,
)
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


def get_model_metas(
    model_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    open_weights: bool | None = None,
    frameworks: Iterable[str] | None = None,
    n_parameters_range: tuple[int | None, int | None] = (None, None),
    use_instructions: bool | None = None,
    zero_shot_on: list[AbsTask] | None = None,
) -> list[ModelMeta]:
    """Load all models' metadata that fit the specified criteria.

    Args:
        model_names: A list of model names to filter by. If None, all models are included.
        languages: A list of languages to filter by. If None, all languages are included.
        open_weights: Whether to filter by models with open weights. If None this filter is ignored.
        frameworks: A list of frameworks to filter by. If None, all frameworks are included.
        n_parameters_range: A tuple of lower and upper bounds of the number of parameters to filter by.
            If (None, None), this filter is ignored.
        use_instructions: Whether to filter by models that use instructions. If None, all models are included.
        zero_shot_on: A list of tasks on which the model is zero-shot. If None this filter is ignored.

    Returns:
        A list of model metadata objects that fit the specified criteria.
    """
    res = []
    model_names = set(model_names) if model_names is not None else None
    languages = set(languages) if languages is not None else None
    frameworks = set(frameworks) if frameworks is not None else None
    for model_meta in MODEL_REGISTRY.values():
        if (model_names is not None) and (model_meta.name not in model_names):
            continue
        if languages is not None:
            if (model_meta.languages is None) or not (
                languages <= set(model_meta.languages)
            ):
                continue
        if (open_weights is not None) and (model_meta.open_weights != open_weights):
            continue
        if (frameworks is not None) and not (frameworks <= set(model_meta.framework)):
            continue
        if (use_instructions is not None) and (
            model_meta.use_instructions != use_instructions
        ):
            continue

        lower, upper = n_parameters_range
        n_parameters = model_meta.n_parameters

        if upper is not None:
            if (n_parameters is None) or (n_parameters > upper):
                continue
            if lower is not None and n_parameters < lower:
                continue

        if zero_shot_on is not None:
            if not model_meta.is_zero_shot_on(zero_shot_on):
                continue
        res.append(model_meta)
    return res


def get_model(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> MTEBModels:
    """A function to fetch and load model object by name.

    !!! note
        This function loads the model into memory. If you only want to fetch the metadata, use [`get_model_meta`](#mteb.get_model_meta) instead.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        **kwargs: Additional keyword arguments to pass to the model loader

    Returns:
        A model object
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer

    meta = get_model_meta(model_name, revision)
    model = meta.load_model(**kwargs)

    # If revision not available in the modelmeta, try to extract it from sentence-transformers
    if hasattr(model, "model") and isinstance(model.model, SentenceTransformer):  # type: ignore
        _meta = _model_meta_from_sentence_transformers(model.model)  # type: ignore
        if meta.revision is None:
            meta.revision = _meta.revision if _meta.revision else meta.revision
        if not meta.similarity_fn_name:
            meta.similarity_fn_name = _meta.similarity_fn_name

    elif isinstance(model, CrossEncoder):
        _meta = _model_meta_from_cross_encoder(model.model)
        if meta.revision is None:
            meta.revision = _meta.revision if _meta.revision else meta.revision

    model.mteb_model_meta = meta  # type: ignore
    return model


def get_model_meta(
    model_name: str, revision: str | None = None, fetch_from_hf: bool = True
) -> ModelMeta:
    """A function to fetch a model metadata object by name.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        fetch_from_hf: Whether to fetch the model from HuggingFace Hub if not found in the registry

    Returns:
        A model metadata object
    """
    if model_name in MODEL_REGISTRY:
        model_meta = MODEL_REGISTRY[model_name]

        if revision and (not model_meta.revision == revision):
            raise ValueError(
                f"Model revision {revision} not found for model {model_name}. Expected {model_meta.revision}."
            )
        return model_meta
    if fetch_from_hf:
        logger.info(
            "Model not found in model registry. Attempting to extract metadata by loading the model ({model_name}) using HuggingFace."
        )
        try:
            meta = _model_meta_from_hf_hub(model_name)
            meta.revision = revision
            return meta
        except RepositoryNotFoundError:
            pass

    not_found_msg = f"Model '{model_name}' not found in MTEB registry"
    not_found_msg += " nor on the Huggingface Hub." if fetch_from_hf else "."

    close_matches = difflib.get_close_matches(model_name, MODEL_REGISTRY.keys())
    model_names_no_org = {mdl: mdl.split("/")[-1] for mdl in MODEL_REGISTRY.keys()}
    if model_name in model_names_no_org:
        close_matches = [model_names_no_org[model_name]] + close_matches

    suggestion = ""
    if close_matches:
        if len(close_matches) > 1:
            suggestion = f" Did you mean: '{close_matches[0]}' or {close_matches[1]}?"
        else:
            suggestion = f" Did you mean: '{close_matches[0]}'?"

    raise KeyError(not_found_msg + suggestion)


def _model_meta_from_hf_hub(model_name: str) -> ModelMeta:
    card = ModelCard.load(model_name)
    card_data = card.data.to_dict()
    frameworks = ["PyTorch"]
    loader = None
    if card_data.get("library_name", None) == "sentence-transformers":
        frameworks.append("Sentence Transformers")
        loader = sentence_transformers_loader
    revision = card_data.get("base_model_revision", None)
    license = card_data.get("license", None)
    return ModelMeta(
        loader=loader,
        name=model_name,
        revision=revision,
        release_date=None,
        languages=None,
        license=license,
        framework=frameworks,  # type: ignore
        training_datasets=None,
        similarity_fn_name=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=None,
    )


def _model_meta_from_cross_encoder(model: CrossEncoder) -> ModelMeta:
    return ModelMeta(
        loader=CrossEncoderWrapper,
        name=model.model.name_or_path,
        revision=model.config._commit_hash,
        release_date=None,
        languages=None,
        framework=["Sentence Transformers"],
        similarity_fn_name=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=None,
        training_datasets=None,
    )


def _model_meta_from_sentence_transformers(model: SentenceTransformer) -> ModelMeta:
    name: str | None = (
        model.model_card_data.model_name
        if model.model_card_data.model_name
        else model.model_card_data.base_model
    )
    embeddings_dim = model.get_sentence_embedding_dimension()
    meta = ModelMeta(
        loader=sentence_transformers_loader,
        name=name,
        revision=model.model_card_data.base_model_revision,
        release_date=None,
        languages=None,
        framework=["Sentence Transformers"],
        similarity_fn_name=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=embeddings_dim,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=None,
        training_datasets=None,
    )
    return meta
