from __future__ import annotations

import logging
from functools import partial

import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import (
    requires_package,
)

logger = logging.getLogger(__name__)


class ColQwen2Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2,
            processor_class=ColQwen2Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


class ColQwen2_5Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2.5 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2_5,
            processor_class=ColQwen2_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


colqwen2 = ModelMeta(
    loader=partial(
        ColQwen2Wrapper,
        model_name="vidore/colqwen2-v1.0",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="vidore/colqwen2-v1.0",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-11-03",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=7200,
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2-v1.0",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

colqwen2_5 = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="vidore/colqwen2.5-v0.2",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="vidore/colqwen2.5-v0.2",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-01-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2.5-v0.2",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

colnomic_7b = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="nomic-ai/colnomic-embed-multimodal-7b",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

COLNOMIC_TRAINING_DATA = {"VDRMultilingual": ["Train"], **COLPALI_TRAINING_DATA}
COLNOMIC_LANGUAGES = [
    "deu-Latn",  # German
    "spa-Latn",  # Spanish
    "eng-Latn",  # English
    "fra-Latn",  # French
    "ita-Latn",  # Italian
]

colnomic_3b = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="nomic-ai/colnomic-embed-multimodal-3b",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="nomic-ai/colnomic-embed-multimodal-3b",
    languages=COLNOMIC_LANGUAGES,
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
)

colnomic_7b = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="nomic-ai/colnomic-embed-multimodal-7b",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    languages=COLNOMIC_LANGUAGES,
    revision="09dbc9502b66605d5be56d2226019b49c9fd3293",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
)
