from __future__ import annotations

from mteb.model_meta import ModelMeta
from mteb.models.bge_models import bge_full_data

lens_d4000 = ModelMeta(
    loader=None,  # TODO: implement this in the future
    name="yibinlei/LENS-d4000",
    languages=None,
    open_weights=True,
    revision="e473b33364e6c48a324796fd1411d3b93670c6fe",
    release_date="2025-01-17",
    n_parameters=int(7.11 * 1e9),
    memory_usage_mb=27125,
    embed_dim=4000,
    license="apache-2.0",
    reference="https://huggingface.co/yibinlei/LENS-d4000",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_full_data,
    max_tokens=32768,
)

lens_d8000 = ModelMeta(
    loader=None,  # TODO: implement this in the future
    name="yibinlei/LENS-d8000",
    languages=None,
    open_weights=True,
    revision="a0b87bd91cb27b6f2f0b0fe22c28026da1d464ef",
    release_date="2025-01-17",
    n_parameters=int(7.11 * 1e9),
    memory_usage_mb=27125,
    embed_dim=8000,
    license="apache-2.0",
    reference="https://huggingface.co/yibinlei/LENS-d8000",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_full_data,
    max_tokens=32768,
)
