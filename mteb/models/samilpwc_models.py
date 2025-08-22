from __future__ import annotations

from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.e5_models import ME5_TRAINING_DATA
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

SAMILPWC_GENAI_TRAINING_DATA = {
    "KorSTS": ["train"],
    "KLUE-TC": ["train"],
    "KLUE-STS": ["train"],
    "KoTripletQA": ["train"],
    "MIRACL": ["train"],
    "KorNLI": ["train"],
}

INSTRUCTION = "Instruct: {instruction}\nQuery: "


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]
        else:
            instruction = instruction[prompt_type]
    return INSTRUCTION.format(instruction=instruction)


def instruct_loader(model_name_or_path, **kwargs):
    model = InstructSentenceTransformerWrapper(
        model_name_or_path,
        revision=kwargs.pop("revision", None),
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        encoder.tokenizer.padding_side = "left"
    return model


samilpwc_expr = ModelMeta(
    loader=partial(
        instruct_loader,
        model_name_or_path="SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
        revision="33358978be40f36491045f9c2a359d38c3f50047",
    ),
    name="SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
    languages=[
        "kor-Hang",
    ],
    open_weights=True,
    revision="33358978be40f36491045f9c2a359d38c3f50047",
    release_date="2025-08-12",
    n_parameters=560_000_000,
    memory_usage_mb=2136,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=514,
    reference="https://huggingface.co/SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="intfloat/multilingual-e5-large-instruct",
    training_datasets={
        **ME5_TRAINING_DATA,
        **SAMILPWC_GENAI_TRAINING_DATA,
    },
)
