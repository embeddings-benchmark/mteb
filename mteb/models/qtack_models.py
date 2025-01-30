from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

mini_gte = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="prdev/mini-gte",
        revision="7fbe6f9b4cc42615e0747299f837ad7769025492",
    ),
    name="prdev/mini-gte",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7fbe6f9b4cc42615e0747299f837ad7769025492",
    release_date="2025-01-28",
    n_parameters=66.3 * 1e6,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/prdev/mini-gte",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # Not released yet
)