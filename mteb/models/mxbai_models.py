from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

mxbai_embed_large_v1 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        revision="990580e27d329c7408b3741ecff85876e128e203",
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: "
        },
    ),
    name="mixedbread-ai/mxbai-embed-large-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision="990580e27d329c7408b3741ecff85876e128e203",
    release_date="2024-03-07",  # initial commit of hf model.
)
