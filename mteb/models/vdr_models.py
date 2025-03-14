from __future__ import annotations

from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return "{instruction}"


languages = [
    "eng_Latn",
    "ita_Latn",
    "fra_Latn",
    "deu_Latn",
    "spa_Latn",
]

vdr_2b_multi_v1 = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="llamaindex/vdr-2b-multi-v1",
        instruction_template=instruction_template,
        max_seq_length=32768,
        apply_instruction_to_passages=True,
    ),
    name="llamaindex/vdr-2b-multi-v1",
    languages=languages,
    open_weights=True,
    revision="2c4e54c8db4071cc61fc3c62f4490124e40c37db",
    release_date="2024-01-08",
    modalities=["text"],  # TODO: integrate with image
    n_parameters=2_000_000_000,
    memory_usage_mb=4213,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/llamaindex/vdr-2b-multi-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Sentence Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/llamaindex/vdr-multilingual-train",
    training_datasets={
        # llamaindex/vdr-multilingual-train
    },
)
