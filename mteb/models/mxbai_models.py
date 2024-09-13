from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.base_wrappers import SentenceTransformerWrapper
from mteb.models.instructions import represent_sentence_instruction

mxbai_embed_large_v1 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        query_instruction=represent_sentence_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="mixedbread-ai/mxbai-embed-large-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision="990580e27d329c7408b3741ecff85876e128e203",
    release_date="2024-03-07",  # initial commit of hf model.
)
