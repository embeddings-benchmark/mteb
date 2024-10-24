from __future__ import annotations

import logging
from functools import partial

from mteb.model_meta import ModelMeta
from .instruct_wrapper import instruct_wrapper

logger = logging.getLogger(__name__)


def gritlm_instruction(instruction: str = "") -> str:
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


gritlm7b = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    open_source=True,
    revision="13f00a0e36500c80ce12870ea513846a066004af",
    release_date="2024-02-15",
)
gritlm8x7b = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-8x7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-8x7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    open_source=True,
    revision="7f089b13e3345510281733ca1e6ff871b5b4bc76",
    release_date="2024-02-15",
)
