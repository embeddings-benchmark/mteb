from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

mixedbread_training_data = {
    # from correspondance:
    # as mentioned in our blog post
    # (https://www.mixedbread.com/blog/mxbai-embed-large-v1#built-for-rag-and-real-world-use-cases:~:text=During%20the%20whole,related%20use%20cases.)
    # We do not train on any data (except the MSMarco training split) of MTEB. We have a strong filtering process to ensure the OOD setting. That's true
    # for all of our models. Keep up the good work and let me know if you have any questions.
    "MSMARCO": [],
}

mxbai_embed_large_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        revision="990580e27d329c7408b3741ecff85876e128e203",
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: "
        },
    ),
    name="mixedbread-ai/mxbai-embed-large-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="990580e27d329c7408b3741ecff85876e128e203",
    release_date="2024-03-07",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=639,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=mixedbread_training_data,
)

mxbai_embed_2d_large_v1 = ModelMeta(
    loader=None,
    name="mixedbread-ai/mxbai-embed-2d-large-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7e639ca8e344af398876ead3b19ec3c0b9068f49",
    release_date="2024-03-04",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/mixedbread-ai/mxbai-embed-2d-large-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=mixedbread_training_data,
)


mxbai_embed_xsmall_v1 = ModelMeta(
    loader=None,
    name="mixedbread-ai/mxbai-embed-xsmall-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2f741ec33328bb57e4704e1238fc59a4a5745705",
    release_date="2024-08-13",  # initial commit of hf model.
    n_parameters=24_100_000,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=384,
    license="apache-2.0",
    reference="https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=mixedbread_training_data,
)
