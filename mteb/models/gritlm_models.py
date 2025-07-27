from __future__ import annotations

import logging
from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.instruct_wrapper import instruct_wrapper

logger = logging.getLogger(__name__)

GRIT_LM_TRAINING_DATA = {
    **E5_MISTRAL_TRAINING_DATA,  # source https://arxiv.org/pdf/2402.09906
    # Note that some models in their ablations also use MEDI2 but not the main GritLM-7B & GritLM-8x7B models
}


def gritlm_instruction(instruction: str = "", prompt_type=None) -> str:
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


gritlm7b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-7B",
    languages=["eng-Latn", "fra-Latn", "deu-Latn", "ita-Latn", "spa-Latn"],
    open_weights=True,
    revision="13f00a0e36500c80ce12870ea513846a066004af",
    release_date="2024-02-15",
    n_parameters=7_240_000_000,
    memory_usage_mb=13813,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/GritLM/GritLM-7B",
    similarity_fn_name="cosine",
    framework=["GritLM", "PyTorch"],
    use_instructions=True,
    training_datasets=GRIT_LM_TRAINING_DATA,
    # section 3.1 "We finetune our final models from Mistral 7B [68] and Mixtral 8x7B [69] using adaptations of E5 [160] and the Tülu 2 data
    public_training_code="https://github.com/ContextualAI/gritlm",
    public_training_data=None,
)

gritlm8x7b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-8x7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-8x7B",
    languages=["eng-Latn", "fra-Latn", "deu-Latn", "ita-Latn", "spa-Latn"],
    open_weights=True,
    revision="7f089b13e3345510281733ca1e6ff871b5b4bc76",
    release_date="2024-02-15",
    n_parameters=57_920_000_000,
    memory_usage_mb=89079,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/GritLM/GritLM-8x7B",
    similarity_fn_name="cosine",
    framework=["GritLM", "PyTorch"],
    use_instructions=True,
    training_datasets=GRIT_LM_TRAINING_DATA,
    # section 3.1 "We finetune our final models from Mistral 7B [68] and Mixtral 8x7B [69] using adaptations of E5 [160] and the Tülu 2 data
    public_training_code="https://github.com/ContextualAI/gritlm",
    public_training_data=None,
)
