"""RuneLab Turkish embedding models."""

from __future__ import annotations

from mteb.models.model_implementations.google_embeddinggemma import (
    EmbeddingGemma2Wrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import MultiVectorWrapper

vor_lateembed_tr_0_6b_v1 = ModelMeta(
    loader=MultiVectorWrapper,
    name="runelab-ai/Vor-LateEmbed-TR-0.6B-v1",
    model_type=["late-interaction"],
    modalities=["image", "text"],
    languages=["tur-Latn", "eng-Latn"],
    open_weights=True,
    revision="f72e14c2997ea7e29d26b932a68b57a8c60335c9",
    release_date="2026-10-12",
    n_parameters=594_321_600,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=2267,
    embed_dim=128,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/runelab-ai/Vor-LateEmbed-TR-0.6B-v1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    adapted_from="perplexity-ai/pplx-embed-v2-late-0.6b",
    # Turkish Wikipedia, news and finance passages (not mteb tasks) plus English MS MARCO replay
    training_datasets={"MSMARCO"},
    public_training_code=None,
    public_training_data=None,
    extra_requirements_groups=["pplx-embed-v2", "image"],
    citation=None,
)

kenning_turkish_legal_v1 = ModelMeta(
    loader=EmbeddingGemma2Wrapper,  # type: ignore[call-arg]
    # the checkpoint ships the text tower only
    loader_kwargs=dict(use_image=False, use_audio=False),
    name="runelab-ai/kenning-turkish-legal-v1",
    model_type=["dense"],
    modalities=["text"],
    languages=["tur-Latn", "eng-Latn"],
    open_weights=True,
    revision="d318f869b1998f947c0058e215a6c8219ed156a0",
    release_date="2026-10-07",
    n_parameters=271_002_624,
    n_embedding_parameters=134_217_728,
    memory_usage_mb=1034,
    embed_dim=[768, 512, 256, 128],
    max_tokens=8192,
    license="apache-2.0",
    reference="https://huggingface.co/runelab-ai/kenning-turkish-legal-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    adapted_from="google/embeddinggemma-2",
    # Turkish court decisions, legislation, synthetic contracts and news summaries; none are mteb tasks
    training_datasets=set(),
    public_training_code=None,
    public_training_data=None,
    extra_requirements_groups=["embeddinggemma"],
    citation=None,
)
