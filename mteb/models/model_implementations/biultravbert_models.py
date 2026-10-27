from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import OutputDType

from .colultravbert_models import COLULTRAVBERT_TRAINING_DATA

biultravbert = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="athrael-soju/BiUltraVBERT",
    model_type=["dense"],
    languages=["eng-Latn", "spa-Latn", "deu-Latn", "fra-Latn", "ita-Latn", "por-Latn"],
    revision="fdb566a46a37a84d740dfa643ac1bf3ddba83c52",
    release_date="2026-10-18",
    modalities=["image", "text"],
    n_parameters=254_679_552,
    n_embedding_parameters=37_847_040,
    memory_usage_mb=486,
    max_tokens=32768,
    embed_dim=[128, 256, 320, 512, 768],
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/athrael-soju/vdr-train-500k",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/athrael-soju/BiUltraVBERT",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=COLULTRAVBERT_TRAINING_DATA,
    adapted_from="athrael-soju/UltraVBERT",
    contacts=["athrael-soju"],
    output_dtypes=OutputDType.BF16,
    extra_requirements_groups=["image"],
)
