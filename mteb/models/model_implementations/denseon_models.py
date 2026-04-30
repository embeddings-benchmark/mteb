from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

denseon_lateon_unsupervised_data = {
    "CQADupstackRetrieval",
    "DBPedia",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
}

denseon_lateon_supervised_data = {
    "FiQA2018",
    "NQ",
    "HotpotQA",
    "MSMARCO",
    "TRECDL2019",
    "TRECDL2020",
    "FEVER",
    "ClimateFEVER",
}

denseon_lateon_citation = r"""@misc{sourty2026denseonlateon,
  title={DenseOn with LateOn: Open State-of-the-Art Single and Multi-Vector Models},
  author={Sourty, Raphael and Chaffin, Antoine and Weller, Orion and Moura Junior, Paulo Roberto and Chatelain, Amelie},
  year={2026},
  howpublished={\url{https://huggingface.co/blog/lightonai/denseon-lateon}},
}"""


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
    public_training_code="",  # We need to add the boilerplates
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
    public_training_code="",  # We need to add the boilerplates
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning",  # As detailed in the BP, the actual training data is proprietary Apache 2 compatible reproduction of this
    training_datasets=denseon_lateon_unsupervised_data | denseon_lateon_supervised_data,
    citation=denseon_lateon_citation,
)
