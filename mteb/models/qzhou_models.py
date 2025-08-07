from __future__ import annotations

import os
from functools import partial
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.bge_models import bge_m3_training_data, bge_full_data, bge_chinese_training_data

def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.passage:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


QZhou_Embedding = ModelMeta(
    loader = partial(
        InstructSentenceTransformerWrapper,
        model_name="Kingsoft-LLM/QZhou-Embedding",
        revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="Kingsoft-LLM/QZhou-Embedding",
    languages=["eng-Latn", "zho-Hans"], 
    open_weights=True,
    revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    release_date="2025-08-01",
    n_parameters=7_070_619_136,
    memory_usage_mb=29070,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets={
        **bge_m3_training_data,
        **bge_chinese_training_data,
        **bge_full_data,
        **E5_MISTRAL_TRAINING_DATA,
        "Shitao/MLDR": ["train"],
        "FreedomIntelligence/Huatuo26M-Lite": ["train"],
        "infgrad/retrieval_data_llm": ["train"],
    },
)
