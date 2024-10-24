from __future__ import annotations

from functools import partial

import torch

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.e5_instruct import e5_loader
from mteb.models.gte_models import gte_loader

STELLA_S2S_PROMPT = "Instruct: Retrieve semantically similar text.\nQuery: "
STELLA_S2P_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "

STELLA_PROMPTS = {
    "query": STELLA_S2P_PROMPT,
    "passage": "",
    "STS": STELLA_S2S_PROMPT,
    "PairClassification": STELLA_S2S_PROMPT,
    "BitextMining": STELLA_S2S_PROMPT,
}

stella_en_400M = ModelMeta(
    loader=partial(
        e5_loader,
        model_name_or_path="dunzhang/stella_en_400M_v5",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        # https://huggingface.co/dunzhang/stella_en_400M_v5/discussions/21#671a6205ac1e2416090f2bf4
        torch_dtype=torch.bfloat16,
    ),
    name="dunzhang/stella_en_400M_v5",
    languages=["eng_Latn"],
    open_source=True,
    revision="1bb50bc7bb726810eac2140e62155b88b0df198f",
    release_date="2024-07-12",
)

stella_en_1_5b = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="dunzhang/stella_en_1.5B_v5",
        revision="d03be74b361d4eb24f42a2fe5bd2e29917df4604",
        model_prompts=STELLA_PROMPTS,
    ),
    name="dunzhang/stella_en_1.5B_v5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d03be74b361d4eb24f42a2fe5bd2e29917df4604",
    release_date="2024-07-12",
)
