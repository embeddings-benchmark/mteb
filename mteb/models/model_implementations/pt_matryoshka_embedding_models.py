from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

PT_MATRYOSHKA_EMBEDDING_CITATION = """
@misc{okamura2026multilingualaveragesmtebptbenchmark,
  archiveprefix = {arXiv},
  author = {Lucas Hideki Takeuchi Okamura and Alexandre Alcoforado and Anna Helena Reali Costa},
  eprint = {2607.04071},
  primaryclass = {cs.CL},
  title = {Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence Encoders},
  url = {https://arxiv.org/abs/2607.04071},
  year = {2026},
}
"""

PT_MATRYOSHKA_EMBEDDING_TRAINING_DATA = {
    "STSBenchmarkMultilingualSTS",  # pt subset
    "Assin2STS",
    "SICK-BR-STS",
    # not in MTEB
    # mldr (pt-triplet subset)
    # assin
    # multilingual-NLI-26lang-2mil7 (pt_anli, pt_fever, pt_ling, pt_mnli, pt_wanli subsets)
}

e5_large_matryoshka_sts_pt = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="iara-project/e5-large-matryoshka-sts-pt",
    revision="85f530106aca75bcb1b8e7b483817b4c942d7810",
    release_date="2026-03-23",
    languages=["por-Latn"],
    n_parameters=559_890_432,
    n_active_parameters_override=None,
    n_embedding_parameters=256_002_048,
    memory_usage_mb=2136,
    max_tokens=514,
    embed_dim=[64, 128, 256, 512, 1024],
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/iara-project/e5-large-matryoshka-sts-pt",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=PT_MATRYOSHKA_EMBEDDING_TRAINING_DATA,
    adapted_from="intfloat/multilingual-e5-large",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=PT_MATRYOSHKA_EMBEDDING_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

BERTimbau_large_matryoshka_sts_pt = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="iara-project/BERTimbau-large-matryoshka-sts-pt",
    revision="691472d1e970512b1f5e975bcc88c7d10b27cf4d",
    release_date="2026-03-23",
    languages=["por-Latn"],
    n_parameters=334_396_416,
    n_active_parameters_override=None,
    n_embedding_parameters=30_509_056,
    memory_usage_mb=1276,
    max_tokens=512,
    embed_dim=[64, 128, 256, 512, 1024],
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/iara-project/BERTimbau-large-matryoshka-sts-pt",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=PT_MATRYOSHKA_EMBEDDING_TRAINING_DATA,
    adapted_from="neuralmind/bert-large-portuguese-cased",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=PT_MATRYOSHKA_EMBEDDING_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

ModBERTBr_matryoshka_sts_pt = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="iara-project/ModBERTBr-matryoshka-sts-pt",
    revision="4d4d51e080ee6b8fc73addc85f89800540fe9be6",
    release_date="2026-04-01",
    languages=["por-Latn"],
    n_parameters=135_497_472,
    n_active_parameters_override=None,
    n_embedding_parameters=25_165_824,
    memory_usage_mb=517,
    max_tokens=512,
    embed_dim=[64, 128, 256, 512, 768],
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/iara-project/ModBERTBr-matryoshka-sts-pt",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=PT_MATRYOSHKA_EMBEDDING_TRAINING_DATA,
    adapted_from="wallacelw/ModBERTBr",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=PT_MATRYOSHKA_EMBEDDING_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
