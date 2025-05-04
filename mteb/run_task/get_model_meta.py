from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models import (
    model_meta_from_cross_encoder,
    model_meta_from_sentence_transformers,
)

empty_model_meta = ModelMeta(
    loader=None,
    name=None,
    revision=None,
    release_date=None,
    languages=None,
    framework=[],
    similarity_fn_name=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=None,
    training_datasets=None,
)


def get_model_meta(model: Encoder | SentenceTransformer | CrossEncoder) -> ModelMeta:
    meta: ModelMeta | None = None
    if hasattr(model, "mteb_model_meta"):
        meta = model.mteb_model_meta  # type: ignore

    if meta is None:
        if isinstance(model, CrossEncoder):
            meta = model_meta_from_cross_encoder(model)
        elif isinstance(model, SentenceTransformer):
            meta = model_meta_from_sentence_transformers(model)
        else:
            meta = empty_model_meta

    # create a copy of the meta to avoid modifying the original object
    meta = deepcopy(meta)
    meta.revision = meta.revision or "no_revision_available"
    meta.name = meta.name or "no_model_name_available"

    return meta


def get_output_folder(
    model_meta: ModelMeta, output_folder: Path | str | None
) -> Path | None:
    """Create output folder for the results."""
    if output_folder is None:
        return None

    model_revision: str = model_meta.revision  # type: ignore
    model_path_name = model_meta.model_name_as_path()

    output_path = Path(output_folder) / model_path_name / model_revision
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
