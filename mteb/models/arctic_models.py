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
    languages=["eng_Latn"],
    open_source=True,
    revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
    release_date="2024-07-08",  # initial commit of hf model.
)
