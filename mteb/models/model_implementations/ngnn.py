"""MTEB registration adapter for the separately distributed NGNN encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from mteb.models.models_protocols import EncoderProtocol

MODEL_NAME = "steffen-negabo/ngnn-general-encoder-v1"
# Canonical evaluation identity: the original public GitHub source commit.
MODEL_REVISION = "d6969c26400944d4f5200ebddfdc04a083fd7b75"
OUTPUT_DIM = 512
HF_REPO_ID = "steffen-negabo/ngnn-general-encoder-v1"
# Later HF distribution of the same weights. See verification/revision_alias.json
# in the public reference repository below.
HF_ARTIFACT_REVISION = "bab7d30011438e52f22be540067c73ca37f462eb"


def load_ngnn_model(
    model_name: str,
    revision: str,
    *,
    base_model: str = "text-embedding-3-large",
    base_url: str = "https://api.openai.com/v1",
    provider_batch_size: int = 100,
    device: str | None = None,
) -> EncoderProtocol:
    """Load the fixed encoder; MTEB supplies and validates its registry identity."""
    if device not in {None, "cpu"}:
        raise ValueError("This NGNN revision uses CPU inference")
    if (
        base_model != "text-embedding-3-large"
        or base_url != "https://api.openai.com/v1"
    ):
        raise ValueError("This NGNN revision requires its registered OpenAI provider")
    from ngnn_encoder import NgnnGeneralEncoder
    from openai import OpenAI

    # Explicit endpoint prevents OPENAI_BASE_URL from changing model identity.
    model = NgnnGeneralEncoder.from_pretrained(
        HF_REPO_ID,
        revision=HF_ARTIFACT_REVISION,
        client=OpenAI(base_url=base_url, max_retries=2),
        provider_batch_size=provider_batch_size,
    )
    return model


ngnn_general_encoder = ModelMeta(
    name="steffen-negabo/ngnn-general-encoder-v1",
    revision=MODEL_REVISION,
    loader=load_ngnn_model,
    loader_kwargs={
        "base_model": "text-embedding-3-large",
        "base_url": "https://api.openai.com/v1",
    },
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
    extra_requirements_groups=["ngnn"],
)
