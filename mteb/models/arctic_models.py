from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

arctic_m_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m-v1.5",
        revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: "
        },
    ),
    name="Snowflake/snowflake-arctic-embed-m-v1.5",
    revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
    release_date="2024-07-08",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    citation="""@misc{merrick2024embeddingclusteringdataimprove,
      title={Embedding And Clustering Your Data Can Improve Contrastive Pretraining}, 
      author={Luke Merrick},
      year={2024},
      eprint={2407.18887},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.18887}, 
    }""",
)


arctic_embed_xs = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-xs",
        revision="742da4f66e1823b5b4dbe6c320a1375a1fd85f9e",
    ),
    name="Snowflake/snowflake-arctic-embed-xs",
    revision="742da4f66e1823b5b4dbe6c320a1375a1fd85f9e",
    release_date="2024-07-08",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=22_600_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=384,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    superseded_by=None,
)


arctic_embed_s = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-s",
        revision="d3c1d2d433dd0fdc8e9ca01331a5f225639e798f",
    ),
    name="Snowflake/snowflake-arctic-embed-s",
    revision="d3c1d2d433dd0fdc8e9ca01331a5f225639e798f",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=32_200_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=384,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-s",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="intfloat/e5-small-unsupervised",
    superseded_by=None,
)


arctic_embed_m = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m",
        revision="cc17beacbac32366782584c8752220405a0f3f40",
    ),
    name="Snowflake/snowflake-arctic-embed-m",
    revision="cc17beacbac32366782584c8752220405a0f3f40",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="intfloat/e5-base-unsupervised",
    superseded_by="Snowflake/snowflake-arctic-embed-m-v1.5",
)

arctic_embed_m_long = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m-long",
        revision="89d0f6ab196eead40b90cb6f9fefec01a908d2d1",
    ),
    name="Snowflake/snowflake-arctic-embed-m-long",
    revision="89d0f6ab196eead40b90cb6f9fefec01a908d2d1",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage=None,
    max_tokens=2048,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="nomic-ai/nomic-embed-text-v1-unsupervised",
    superseded_by=None,
)


arctic_embed_l = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-l",
        revision="9a9e5834d2e89cdd8bb72b64111dde496e4fe78c",
    ),
    name="Snowflake/snowflake-arctic-embed-l",
    revision="9a9e5834d2e89cdd8bb72b64111dde496e4fe78c",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng_Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-l",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="intfloat/e5-base-unsupervised",
    superseded_by=None,
)
