from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.base_wrappers import SentenceTransformerWrapper
from mteb.models.instructions import gte_instruction

stella_en_400M_v5 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="dunzhang/stella_en_400M_v5",
        query_instruction=gte_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="dunzhang/stella_en_400M_v5",
    languages=None,
    open_source=True,
    revision="1bb50bc7bb726810eac2140e62155b88b0df198f",
    release_date="2024-07-12",  # initial commit of hf model.
)

stella_en_400M_v5 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="dunzhang/stella_en_1.5B_v5",
        query_instruction=gte_instruction,
        corpus_instruction=None,
    ),  # type: ignore
    name="dunzhang/stella_en_1.5B_v5",
    languages=None,
    open_source=True,
    revision="d03be74b361d4eb24f42a2fe5bd2e29917df4604",
    release_date="2024-07-12",  # initial commit of hf model.
)
