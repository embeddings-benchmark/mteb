from __future__ import annotations

import difflib
import logging
import warnings
from typing import TYPE_CHECKING, Any

from mteb.models import (
    ModelMeta,
)
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mteb.abstasks import AbsTask
    from mteb.models import (
        MTEBModels,
    )

logger = logging.getLogger(__name__)


def get_model_metas(
    model_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    open_weights: bool | None = None,
    frameworks: Iterable[str] | None = None,
    n_parameters_range: tuple[int | None, int | None] = (None, None),
    use_instructions: bool | None = None,
    zero_shot_on: list[AbsTask] | None = None,
    model_types: Iterable[str] | None = None,
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
        model_types: A list of model types to filter by. If None, all model types are included.

    Returns:
        A list of model metadata objects that fit the specified criteria.
    """
    res = []
    model_names = set(model_names) if model_names is not None else None
    languages = set(languages) if languages is not None else None
    frameworks = set(frameworks) if frameworks is not None else None
    model_types_set = set(model_types) if model_types is not None else None
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
        if model_types_set is not None and not model_types_set.intersection(
            model_meta.model_type
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
    model_name: str,
    revision: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> MTEBModels:
    """A function to fetch and load model object by name.

    !!! note
        This function loads the model into memory. If you only want to fetch the metadata, use [`get_model_meta`](#mteb.get_model_meta) instead.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        device: Device used to load the model
        **kwargs: Additional keyword arguments to pass to the model loader

    Returns:
        A model object
    """
    meta = get_model_meta(model_name, revision)
    model = meta.load_model(device=device, **kwargs)

    if kwargs:
        logger.info(
            f"Model '{model_name}' loaded with additional arguments: {list(kwargs.keys())}"
        )
        meta = meta.model_copy(deep=True)
        meta.loader_kwargs |= kwargs

    model.mteb_model_meta = meta  # type: ignore[misc]
    return model


_MODEL_RENAMES: dict[str, str] = {
    "bm25s": "baseline/bm25s",
}


def get_model_meta(
    model_name: str,
    revision: str | None = None,
    fetch_from_hf: bool = True,
    fill_missing: bool = False,
) -> ModelMeta:
    """A function to fetch a model metadata object by name.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        fetch_from_hf: Whether to fetch the model from HuggingFace Hub if not found in the registry
        fill_missing: Computes missing attributes from the metadata including number of parameters and memory usage.

    Returns:
        A model metadata object
    """
    if model_name in _MODEL_RENAMES:
        new_name = _MODEL_RENAMES[model_name]
        msg = f"The model '{model_name}' has been renamed to '{new_name}'. To prevent this warning use the new name."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        model_name = new_name

    if model_name in MODEL_REGISTRY:
        model_meta = MODEL_REGISTRY[model_name]

        if revision and (not model_meta.revision == revision):
            raise ValueError(
                f"Model revision {revision} not found for model {model_name}. Expected {model_meta.revision}."
            )

        if fill_missing and fetch_from_hf:
            original_meta_dict = model_meta.model_dump()
            new_meta = ModelMeta.from_hub(model_name)
            new_meta_dict = new_meta.model_dump(exclude_none=True)

            updates = {
                k: v
                for k, v in new_meta_dict.items()
                if original_meta_dict.get(k) is None
            }

            if updates:
                return model_meta.model_copy(update=updates)
        return model_meta

    if fetch_from_hf:
        logger.info(
            f"Model not found in model registry. Attempting to extract metadata by loading the model ({model_name}) using HuggingFace."
        )
        meta = ModelMeta.from_hub(model_name, revision)
        return meta

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
