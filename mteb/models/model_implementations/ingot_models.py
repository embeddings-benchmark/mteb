"""ModelMeta for Ingot-8B-R3 (JCorners/Ingot-8B-R3).

Ingot-8B-R3 is built on Qwen3-Embedding-8B with a proprietary routing
framework. Different specialists activate at inference time from input
content alone (no task metadata required). The routing framework is
patent-pending; weights are available via gated access on HuggingFace.

Access: https://huggingface.co/JCorners/Ingot-8B-R3 (Request Access button)
Contact for verification: corp@voxell.ai
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from mteb.models.models_protocols import EncoderProtocol


def _instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


def ingot_8b_r3_loader(
    model_name_or_path: str, revision: str, **kwargs
) -> EncoderProtocol:
    """Load Ingot-8B-R3 from the gated HuggingFace repo.

    This model is gated with manual approval. Use the Request Access button
    at https://huggingface.co/JCorners/Ingot-8B-R3 or contact corp@voxell.ai.
    Once access is granted, this loader instantiates the model via the standard
    SentenceTransformer interface (trust_remote_code=True required for the
    routing layer).
    """
    model = InstructSentenceTransformerModel(
        model_name_or_path,
        revision=revision,
        instruction_template=_instruction_template,
        apply_instruction_to_passages=False,
        trust_remote_code=True,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        encoder.tokenizer.padding_side = "left"
    return model


Ingot_8B_R3 = ModelMeta(
    loader=ingot_8b_r3_loader,
    name="JCorners/Ingot-8B-R3",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=False,
    revision="09e3bdb941f1faec98c0326c7c8e22635189d83e",
    release_date="2026-05-21",
    n_parameters=7_567_295_488,
    n_embedding_parameters=None,
    memory_usage_mb=14433,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/JCorners/Ingot-8B-R3",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(),
)
