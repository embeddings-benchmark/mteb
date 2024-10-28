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
    embed_dim=256,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5",
    similarity_fn_name="cosine_similarity",
    use_instructions=False,
)
