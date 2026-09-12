"""MTEB registration adapter for the separately distributed NGNN encoder."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, Any

from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.models.models_protocols import EncoderProtocol

MODEL_NAME = "steffen-negabo/ngnn-general-encoder-v1"
MODEL_REVISION = "d6969c26400944d4f5200ebddfdc04a083fd7b75"
OUTPUT_DIM = 512
PACKAGE_DISTRIBUTION = "ngnn-encoder"
PACKAGE_VERSION = "0.1.0"
PACKAGE_INSTALL_URL = (
    "https://github.com/steffen181/frozen-ngnn-api-modesl/releases/download/"
    "v0.1.0/ngnn_encoder-0.1.0-py3-none-any.whl"
    "#sha256=1ea29b3717175aacc2b6f1465d456201446a2c4eec835428526807bd6c7833e3"
)

HF_REPO_ID: str | None = "steffen-negabo/ngnn-general-encoder-v1"
HF_ARTIFACT_REVISION: str | None = "bab7d30011438e52f22be540067c73ca37f462eb"


def _load_encoder_class() -> Any:  # noqa: ANN401 -- package class is imported lazily
    """Import the pinned inference package only when a model is requested."""
    install = f"pip install '{PACKAGE_INSTALL_URL}'"
    try:
        installed = importlib.metadata.version(PACKAGE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"Install {PACKAGE_DISTRIBUTION}=={PACKAGE_VERSION} with: {install}"
        ) from None
    if installed != PACKAGE_VERSION:
        raise RuntimeError(
            f"This adapter requires {PACKAGE_DISTRIBUTION}=={PACKAGE_VERSION}; "
            f"found {installed}. Reinstall with: {install}"
        )
    try:
        from ngnn_encoder import NgnnGeneralEncoder
    except ImportError:
        raise RuntimeError(
            f"Could not import {PACKAGE_DISTRIBUTION}=={PACKAGE_VERSION}; "
            f"reinstall with: {install}"
        ) from None
    return NgnnGeneralEncoder


def load_ngnn_model(
    model_name: str = MODEL_NAME,
    revision: str | None = MODEL_REVISION,
    *,
    artifact_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    client: Any | None = None,  # noqa: ANN401 -- accepts OpenAI-compatible clients
    provider_batch_size: int = 100,
    device: str | None = None,
    **kwargs: Any,
) -> EncoderProtocol:
    """Load the fixed model through the reviewed ``ngnn-encoder`` package."""
    if "api_key" in kwargs:
        raise ValueError("Use OPENAI_API_KEY; MTEB serializes model keyword arguments")
    if kwargs:
        raise ValueError("Unsupported NGNN model keyword arguments")
    if model_name != MODEL_NAME:
        raise ValueError("Unsupported NGNN model name")
    if revision not in {None, MODEL_REVISION}:
        raise ValueError("Unsupported NGNN model revision")
    if device not in {None, "cpu"}:
        raise ValueError("This NGNN revision uses CPU inference")
    if artifact_path is None and (HF_REPO_ID is None or HF_ARTIFACT_REVISION is None):
        raise RuntimeError(
            "NGNN Hub loading is not yet published; pass artifact_path explicitly"
        )

    encoder_class = _load_encoder_class()
    if artifact_path is not None:
        model = encoder_class(
            artifact_path,
            client=client,
            provider_batch_size=provider_batch_size,
        )
    else:
        model = encoder_class.from_pretrained(
            HF_REPO_ID,
            revision=HF_ARTIFACT_REVISION,
            cache_dir=cache_dir,
            client=client,
            provider_batch_size=provider_batch_size,
        )
    model.mteb_model_meta = ngnn_general_encoder.model_copy(deep=True)
    return model


ngnn_general_encoder = ModelMeta(
    name=MODEL_NAME,
    revision=MODEL_REVISION,
    loader=load_ngnn_model,
    release_date="2026-09-06",
    languages=["eng-Latn"],
    # The local dictionary has 1,572,864 values plus 512 fixed unit scales.
    # Totals include the proprietary OpenAI base and are unknown.
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8191,
    embed_dim=OUTPUT_DIM,
    license="mit",  # Published compressor/code; OpenAI API terms remain separate.
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=["API", "PyTorch"],  # Encoding still requires OpenAI API access.
    reference="https://github.com/steffen181/frozen-ngnn-api-modesl",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from="openai/text-embedding-3-large",
    modalities=["text"],
    model_type=["dense"],
    contacts=["steffen181"],
    extra_requirements_groups=["openai"],
)
