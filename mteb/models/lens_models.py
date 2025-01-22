from __future__ import annotations

from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.instruct_wrapper import instruct_wrapper

lens_d4000 = ModelMeta(
    loader=None,  # TODO: implement this in the future
    name="yibinlei/LENS-d4000",
    languages=None,
    open_weights=True,
    revision="e473b33364e6c48a324796fd1411d3b93670c6fe",
    release_date="2025-01-17",
    n_parameters=int(7.11 * 1e9),
    embed_dim=4000,
    license="apache-2.0",
    reference="https://huggingface.co/yibinlei/LENS-d4000",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
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
    embed_dim=8000,
    license="apache-2.0",
    reference="https://huggingface.co/yibinlei/LENS-d8000",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    max_tokens=32768,
)
