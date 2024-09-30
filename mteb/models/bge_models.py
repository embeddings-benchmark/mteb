from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.base_wrappers import SentenceTransformerWrapper
from mteb.models.instructions import represent_sentence_instruction


def query_instruction(instruction: str) -> str:
    return f"<instruct>{instruction}\n<query> "


bge_base_en_v1_5 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="BAAI/bge-base-en-v1.5",
        query_instruction=represent_sentence_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="BAAI/bge-large-en-v1.5",
        query_instruction=represent_sentence_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
)

bge_multilingual_gemma2 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="BAAI/bge-multilingual-gemma2",
        query_instruction=query_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="BAAI/bge-multilingual-gemma2",
    languages=["eng_Latn", "zho_Hans", "jpn_Latn", "kor_Hang", "fra_Latn"],
    open_source=True,
    revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    release_date="2024-06-25",  # initial commit of hf model.
)
