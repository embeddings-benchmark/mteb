"""Zhinao Chinese ModernBert embedding models by Qihoo360"""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

zhinao_modernbert_zh_datasets = {
    "BQ",
    "LCQMC",
    "PAWSX",
    "STS-B",
    "DuRetrieval",
    "AFQMC",
    "Cmnli",
    "Ocnli",
    "ThuNEWS",
    "CMedQAv2",
    "MMarco",
    "T2Retrieval",
    "JDReview"
}

zhinao_chinesemodernbert_embedding = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs={},
    name='qihoo360/Zhinao-ChineseModernBert-Embedding',
    revision='a476584b3d44d0af3d998c948f38bfc6b97e40a7',
    release_date="2026-03-18",
    languages=["zho-Hans"],
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=116480256,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=['Sentence Transformers', 'PyTorch'],
    reference="https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=zhinao_modernbert_zh_datasets,
    adapted_from=None,
    superseded_by=None,
    modalities=['text'],
    model_type=['dense'],
    citation=None,
    contacts=None,
)
