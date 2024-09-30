from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.base_wrappers import SentenceTransformerWrapper
from mteb.models.instructions import gte_instruction

Linq_Embed_Mistral = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="Linq-AI-Research/Linq-Embed-Mistral",
        query_instruction=gte_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="Linq-AI-Research/Linq-Embed-Mistral",
    languages=None,
    open_source=True,
    revision="0c1a0b0589177079acc552433cad51d7c9132379",
    release_date="2024-05-29",  # initial commit of hf model.
)
