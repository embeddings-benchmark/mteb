from __future__ import annotations

from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        # https://github.com/facebookresearch/ReasonIR/blob/0aac96269e455965949df16520fab72da68ffc22/evaluation/bright/configs/reasonir/economics.json#L3
        f"<|user|>\n{instruction}<|embed|>\n"
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else "<|embed|>\n"
    )


REASONIR_TRAINING_DATA = {
    # source, section D: https://arxiv.org/pdf/2504.20595
    "MSMARCO": ["train"],
    "NQ": ["train"],
    "FEVER": ["train"],
    "HotpotQA": ["train"],
    "MIRACLRetrieval": ["train"],
    "MrTidyRetrieval": ["train"],
    "T2Reranking": ["train"],
    "DuRetrieval": ["train"],
    "QuoraRetrieval": ["train"],
}

ReasonIR_8B = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="ReasonIR/ReasonIR-8B",
        revision="c3d0690370ff4a8c3d3882d8dfa85c43650034fa",
        instruction_template=instruction_template,
        trust_remote_code=True,
    ),
    name="ReasonIR/ReasonIR-8B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c3d0690370ff4a8c3d3882d8dfa85c43650034fa",
    release_date="2025-04-29",
    n_parameters=7_500_000_000,
    memory_usage_mb=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=131072,
    reference="https://huggingface.co/ReasonIR/ReasonIR-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=REASONIR_TRAINING_DATA,
    public_training_code="https://github.com/facebookresearch/ReasonIR/tree/main/training",
    public_training_data="https://huggingface.co/datasets/reasonir/reasonir-data",
)
