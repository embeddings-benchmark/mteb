from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}

spice = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="iamgroot42/spice",
        revision="1d0fdb2b5d7aed81bba960cd8c85671674d49bdc",
        model_prompts=model_prompts,
    ),
    name="iamgroot42/spice",
    languages=["eng_Latn"],
    open_weights=True,
    revision="1d0fdb2b5d7aed81bba960cd8c85671674d49bdc",
    release_date="2025-01-27",  # initial commit of hf model.
    n_parameters=24_000_000,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/iamgroot42/spice",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
