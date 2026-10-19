from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import MultiVectorWrapper
from mteb.types import OutputDType

from .colpali_models import COLPALI_TRAINING_DATA

COLULTRAVBERT_TRAINING_DATA = {
    *COLPALI_TRAINING_DATA,
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "VisRAGRetArxivQA",
    "VisRAGRetChartQA",
    "VisRAGRetInfoVQA",
    "VisRAGRetMPDocVQA",
    "VisRAGRetPlotQA",
    "VisRAGRetSlideVQA",
    "VDRMultilingualRetrieval",
    "docmatix-ir",
    "wiki-ss-nq",
    "NQ",
}

colultravbert = ModelMeta(
    loader=MultiVectorWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="athrael-soju/ColUltraVBERT",
    model_type=["late-interaction"],
    languages=["eng-Latn", "spa-Latn", "deu-Latn", "fra-Latn", "ita-Latn", "por-Latn"],
    revision="0869b8d18e11808c7727831f0ffa2522be0027a2",
    release_date="2026-10-17",
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
    reference="https://huggingface.co/athrael-soju/ColUltraVBERT",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLULTRAVBERT_TRAINING_DATA,
    adapted_from="athrael-soju/UltraVBERT",
    contacts=["athrael-soju"],
    output_dtypes=OutputDType.BF16,
    extra_requirements_groups=["image"],
)
