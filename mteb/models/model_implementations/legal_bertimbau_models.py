from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

LEGAL_BERTIMBAU_TRAINING_DATA = {
    "STSBenchmarkMultilingualSTS",  # pt subset
    "Assin2STS",
    # not in MTEB
    # assin
}

Legal_BERTimbau_sts_large_ma_v3 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="rufimelo/Legal-BERTimbau-sts-large-ma-v3",
    revision="24099071e7f31e022b603620f498cb87d6c78a85",
    release_date="2022-09-21",
    languages=["por-Latn"],
    n_parameters=334_396_928,
    n_active_parameters_override=None,
    n_embedding_parameters=30_509_056,
    memory_usage_mb=1276,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/rufimelo/Legal-BERTimbau-sts-large-ma-v3",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=LEGAL_BERTIMBAU_TRAINING_DATA,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
