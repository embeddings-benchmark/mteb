"""Implementation of the Stable Static Embedding (SSE) model family by Rikka-Botan.

Collection: https://huggingface.co/collections/RikkaBotan/sse-stable-static-embedding

SSE is a token-pooling static embedding architecture (EmbeddingBag + SeparableDyT)
trained with Matryoshka and Multiple-Negatives-Ranking losses.
"""

import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

SSE_CITATION = """@misc{rikkabotan2026sse,
  author       = {Rikka Botan},
  title        = {Stable Static Embedding (SSE): Fast Retrieval with Matryoshka Representation Learning},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/collections/RikkaBotan/sse-stable-static-embedding},
}"""

# Datasets that overlap with MTEB tasks (best-effort intersection of the
# HF training datasets listed on each model card with mteb's task registry).
_SSE_EN_TRAINING_DATASETS = {
    # sentence-transformers/hotpotqa
    "HotpotQA",
    # sentence-transformers/miracl
    "MIRACLRetrieval",
    # sentence-transformers/mr-tydi
    "MrTidyRetrieval",
    # tomaarsen/natural-questions-hard-negatives
    "NQ",
    # nthakur/swim-ir-monolingual, sentence-transformers/{squad,trivia-qa-triplet,
    # all-nli,pubmedqa,s2orc,paq} are listed on the cards but do not have a direct
    # mteb counterpart (or are not used as evaluation tasks), so they are omitted.
    "SCIDOCS",
    "MultiLongDocRetrieval",
}

_SSE_JA_TRAINING_DATASETS = {
    # tomaarsen/NanoBEIR-ja is a translation of NanoBEIR — overlaps with the
    # Nano* tasks but those are mostly English; we mark only the JA tasks that
    # are likely covered by hotchpotch/sentence_transformer_japanese.
    "MIRACLJaRetrievalLite",
    "MrTidyRetrieval",
}


stable_static_embedding_mrl_en = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
    revision="176cf7f95971893536afabb72ed1a9dc491e23df",
    release_date="2026-02-08",
    languages=["eng-Latn"],
    n_parameters=15_628_800,
    n_embedding_parameters=15_628_800,
    memory_usage_mb=60,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_EN_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


stable_static_embedding_mrl_en_v2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en-v2",
    revision="627f9d1a07a4e50a206a21868cf49854f7188a6d",
    release_date="2026-05-12",
    languages=["eng-Latn"],
    n_parameters=15_628_800,
    n_embedding_parameters=15_628_800,
    memory_usage_mb=60,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_EN_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    adapted_from="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
    superseded_by=None,
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


stable_static_embedding_mrl_ja = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-ja",
    revision="c58315fc2bef15c505fd8f7340056f603b4f1dbd",
    release_date="2026-02-28",
    languages=["jpn-Jpan"],
    n_parameters=16_778_752,
    n_embedding_parameters=16_778_752,
    memory_usage_mb=64,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/RikkaBotan/stable-static-embedding-fast-retrieval-mrl-ja",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_JA_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


stable_static_embedding_mrl_bilingual_ja_en = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-bilingual-ja-en",
    revision="88ffe32551ea1d75170947bf6c681700dd7495a1",
    release_date="2026-02-27",
    languages=["eng-Latn", "jpn-Jpan"],
    n_parameters=49_597_440,
    n_embedding_parameters=49_597_440,
    memory_usage_mb=189,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/RikkaBotan/stable-static-embedding-fast-retrieval-mrl-bilingual-ja-en",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_EN_TRAINING_DATASETS | _SSE_JA_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


quantized_stable_static_embedding_mrl_en = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-en",
    revision="48de1fc899574177a7c2d0d4ae90bb902190935c",
    release_date="2026-02-21",
    languages=["eng-Latn"],
    n_parameters=15_628_800,
    n_embedding_parameters=15_628_800,
    # ~7.9 MB packed 4-bit embedding + tiny safetensors rest
    memory_usage_mb=8,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-en",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_EN_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    adapted_from="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


quantized_stable_static_embedding_mrl_ja = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-ja",
    revision="620c78a1b2d3203b24104849ac79dd94cac8eeea",
    release_date="2026-02-28",
    languages=["jpn-Jpan"],
    n_parameters=16_778_752,
    n_embedding_parameters=16_778_752,
    memory_usage_mb=9,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-ja",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_JA_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    adapted_from="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-ja",
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)


quantized_stable_static_embedding_mrl_bilingual_ja_en = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-bilingual-ja-en",
    revision="0a6a0e901f477ab513ee1bc4931ef322937bb742",
    release_date="2026-03-04",
    languages=["eng-Latn", "jpn-Jpan"],
    n_parameters=49_597_440,
    n_embedding_parameters=49_597_440,
    memory_usage_mb=25,
    max_tokens=np.inf,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-bilingual-ja-en",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSE_EN_TRAINING_DATASETS | _SSE_JA_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    adapted_from="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-bilingual-ja-en",
    contacts=["Rikka-Botan"],
    citation=SSE_CITATION,
)
