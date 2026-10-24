"""Model definitions for kazalbrur's Bengali/Banglish sentence encoders."""

from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .e5_models import ME5_TRAINING_DATA, model_prompts

BANGLA_EMBED_TRAINING_DATA = {
    # NLI polish stage: XNLI-bn, IndicXNLI-bn and MNLI-en
    "XNLI",
    "XNLIV2",
    "IndicXnliPairClassification",
    # contrastive stage includes a Bengali translation of MS MARCO
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
} | ME5_TRAINING_DATA  # inherited from the intfloat/multilingual-e5-small backbone

bangla_embed_e5_small_banglish = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="kazalbrur/bangla-embed-e5-small-banglish",
    model_type=["dense"],
    languages=["ben-Beng", "ben-Latn", "eng-Latn"],
    open_weights=True,
    revision="bf4c0da806dfb707a39019e16a9c24b215b30e7d",
    release_date="2026-06-23",
    n_parameters=118_048_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=450,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/kazalbrur/bangla-embed-e5-small-banglish",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BANGLA_EMBED_TRAINING_DATA,
    adapted_from="intfloat/multilingual-e5-small",
    superseded_by=None,
    citation=None,
)
