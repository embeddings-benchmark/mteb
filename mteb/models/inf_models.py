from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

inf_retriever_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="infly/inf-retriever-v1",
        revision="d2d074546028c0012b5cc6af78c4fac24896e67f",
        trust_remote_code=True,
    ),
    name="infly/inf-retriever-v1",
    languages=["eng_Latn", "zho_Hans"],
    open_weights=True,
    revision="d2d074546028c0012b5cc6af78c4fac24896e67f",
    release_date="2024-12-24",  # initial commit of hf model.
    n_parameters=7_069_121_024,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=131_072,
    reference="https://huggingface.co/infly/inf-retriever-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-Qwen2-7B-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
