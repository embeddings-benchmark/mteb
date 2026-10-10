"""UIBK-Embedding-v1: a late-interaction (ColBERT) head and a dense head in one encoder.

One 179M-parameter English encoder (ModernBERT-base), saved as a PyLate ColBERT model. Every text becomes a list of
128-dim vectors (3 vectors carrying the dense sentence vector + one per token), and plain MaxSim over them equals
token MaxSim + 0.75 x dense cosine. It therefore runs through the standard PyLate wrapper (PLAID index) without
any special code here; the two custom modules live in the model repository (trust_remote_code).
"""

from mteb.models.model_implementations.pylate_models import (
    MultiVectorModel,
    denseon_lateon_supervised_data,
    denseon_lateon_unsupervised_data,
)
from mteb.models.model_meta import ModelMeta

NAME = "DataScience-UIBK/uibk-embedding-v1"  # Hugging Face repo id
REVISION = "b18df82e77306006a1dd0162199f6dfeacb5492e"  # Hub commit hash

uibk_embedding_v1 = ModelMeta(
    loader=MultiVectorModel,
    loader_kwargs=dict(
        trust_remote_code=True,  # branched_transformer.py / hybrid_tokens.py in the model repo
        # the precision of every reported number; a string, so that this file does not import torch
        model_kwargs={"torch_dtype": "bfloat16"},
        # FastPlaid picks its scoring batch from the free GPU memory. PyLate's fixed default (2**18) runs out of
        # memory on the corpora of several million documents, because this model indexes documents of up to 2048
        # tokens. Batching only: the scores do not depend on it.
        index_kwargs={"batch_size": "auto"},
    ),
    name=NAME,
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    open_weights=True,
    revision=REVISION,
    public_training_code=None,
    # LightOn's public pre-training and fine-tuning collections (the data behind DenseOn / LateOn), BEIR test queries
    # removed; plus pairs mined from the same pre-training collection
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning",
    release_date="2026-10-10",
    n_parameters=179_017_728,
    n_embedding_parameters=38_684_160,
    memory_usage_mb=683,
    max_tokens=2048,
    embed_dim=128,
    license="apache-2.0",
    similarity_fn_name="MaxSim",
    framework=["PyLate", "ColBERT", "safetensors", "Sentence Transformers"],
    reference=f"https://huggingface.co/{NAME}",
    use_instructions=False,
    adapted_from="answerdotai/ModernBERT-base",
    superseded_by=None,
    # same training collections as lightonai/LateOn (pre-training mixture + supervised fine-tuning set)
    training_datasets=denseon_lateon_unsupervised_data | denseon_lateon_supervised_data,
    citation=None,
    extra_requirements_groups=["pylate"],
)
