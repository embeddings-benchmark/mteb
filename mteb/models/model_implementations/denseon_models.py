from mteb.models.model_implementations.pylate_models import (
    denseon_lateon_citation,
    denseon_lateon_supervised_data,
    denseon_lateon_unsupervised_data,
    mdenseon_mlateon_citation,
    mdenseon_mlateon_code_data,
    mdenseon_mlateon_organic_data,
)
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

lightonai__denseon_unsupervised = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="lightonai/DenseOn-unsupervised",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0edbd55684eb782bce55ee74c95b25c97cbe7f43",
    release_date="2026-04-21",
    n_parameters=149014272,
    n_embedding_parameters=38682624,
    memory_usage_mb=568,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/DenseOn-unsupervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "safetensors",
    ],
    use_instructions=False,
    adapted_from="answerdotai/ModernBERT-base",
    superseded_by=None,
    public_training_code="https://github.com/lightonai/mdenseon-mlateon/blob/main/scripts/pretrain/english_dense.py",
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated",  # As detailed in the BP, the actual training data is proprietary Apache 2 compatible reproduction of this
    training_datasets=denseon_lateon_unsupervised_data,
    citation=denseon_lateon_citation,
)


lightonai__denseon = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="lightonai/DenseOn",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="41b4bba613f8ef85c61a69ef7d66410e1478567d",
    release_date="2026-04-21",
    n_parameters=149014272,
    n_embedding_parameters=38682624,
    memory_usage_mb=568,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/DenseOn",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "safetensors",
    ],
    use_instructions=False,
    adapted_from="lightonai/DenseOn-unsupervised",
    superseded_by=None,
    public_training_code="https://github.com/lightonai/mdenseon-mlateon/blob/main/scripts/finetune/english_dense.py",
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning",  # As detailed in the BP, the actual training data is proprietary Apache 2 compatible reproduction of this
    training_datasets=denseon_lateon_unsupervised_data | denseon_lateon_supervised_data,
    citation=denseon_lateon_citation,
)


# English plus the eight translate-train target languages
mdenseon_mlateon_languages = [
    "ara-Arab",
    "deu-Latn",
    "eng-Latn",
    "fra-Latn",
    "ita-Latn",
    "nob-Latn",
    "por-Latn",
    "spa-Latn",
    "swe-Latn",
]

# Code languages covered by the LateOn-Code fine-tuning data
mdenseon_mlateon_code_languages = [
    "python-Code",
    "go-Code",
    "java-Code",
    "javascript-Code",
    "ruby-Code",
    "php-Code",
]

lightonai__mdenseon_unsupervised = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="lightonai/mDenseOn-unsupervised",
    model_type=["dense"],
    languages=mdenseon_mlateon_languages,
    open_weights=True,
    revision="5fc35cd49da5c561a8c1db20d7f349c036455919",
    release_date="2026-07-30",
    n_parameters=306939648,
    n_embedding_parameters=196608000,
    memory_usage_mb=1171,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/mDenseOn-unsupervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "safetensors",
    ],
    use_instructions=False,
    adapted_from="jhu-clsp/mmBERT-base",
    superseded_by=None,
    public_training_code="https://github.com/lightonai/mdenseon-mlateon/blob/main/scripts/pretrain/multilingual_dense.py",
    public_training_data="https://huggingface.co/datasets/lightonai/multilingual-embeddings-pre-training-curated",  # As detailed in the BP, the actual training data is a proprietary Apache 2 compatible reproduction of this
    training_datasets=denseon_lateon_unsupervised_data,
    citation=mdenseon_mlateon_citation,
)


lightonai__mdenseon = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="lightonai/mDenseOn",
    model_type=["dense"],
    languages=mdenseon_mlateon_languages + mdenseon_mlateon_code_languages,
    open_weights=True,
    revision="d61878b089bba2156fe029d3bb8ccaad368a6249",
    release_date="2026-07-30",
    n_parameters=306939648,
    n_embedding_parameters=196608000,
    memory_usage_mb=1171,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/mDenseOn",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "safetensors",
    ],
    use_instructions=False,
    adapted_from="lightonai/mDenseOn-unsupervised",
    superseded_by=None,
    public_training_code="https://github.com/lightonai/mdenseon-mlateon/blob/main/scripts/finetune/multilingual_dense.py",
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-multilingual-unfiltered",  # Filtered per-language versions are linked in the collection: https://huggingface.co/collections/lightonai/mdenseon-and-mlateon
    training_datasets=denseon_lateon_unsupervised_data
    | denseon_lateon_supervised_data
    | mdenseon_mlateon_organic_data
    | mdenseon_mlateon_code_data,
    citation=mdenseon_mlateon_citation,
)
