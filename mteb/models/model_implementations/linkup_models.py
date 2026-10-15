from mteb.models.model_implementations.pylate_models import (
    denseon_lateon_supervised_data,
    denseon_lateon_unsupervised_data,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SparseEncoderWrapper

linkup_sparseup_embed_v1 = ModelMeta(
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="Linkup-Platform/linkup-sparseup-embed-v1",
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="08314498d4f6a3a205b930ab9f27001404ea94b8",
    release_date="2026-09-16",
    n_parameters=188391300,
    n_embedding_parameters=38684160,
    memory_usage_mb=719,
    max_tokens=512,  # documents are truncated to 512 tokens and queries to 128 by the model itself
    embed_dim=50370,
    license="apache-2.0",
    reference="https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=False,
    adapted_from="lightonai/LateOn-unsupervised",
    superseded_by=None,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning",
    # fine-tuned from LateOn-unsupervised on the DenseOn/LateOn fine-tuning mixture
    training_datasets=denseon_lateon_unsupervised_data | denseon_lateon_supervised_data,
)
