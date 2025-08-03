from __future__ import annotations

import os
from functools import partial
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

def instruction_template(instruction: str) -> str:
    if not instruction: return ''

    return 'Instruct: {}\nQuery: '.format(instruction)


def qzhou_instruct_loader(model_name, **kwargs):
    model = InstructSentenceTransformerWrapper(
        model_name,
        revision=kwargs.pop("model_revision", None),
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    encoder.tokenizer.padding_side = "left"
    return model


QZhou_Embedding = ModelMeta(
    loader = partial(
        qzhou_instruct_loader,
        model_name="Kingsoft-LLM/QZhou-Embedding",
        model_revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
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
    public_training_data="https://huggingface.co/cfli/datasets",
    training_datasets={"bge-e5data": ["train"], "bge-full-data": ['train']},
)
