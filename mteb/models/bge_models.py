from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

bge_small_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
)
