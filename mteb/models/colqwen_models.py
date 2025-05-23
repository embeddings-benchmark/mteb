from __future__ import annotations

import logging
from functools import partial
from typing import Literal

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import ColPaliEngineWrapper

logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class ColQwen2Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self, model_name: str = "vidore/colqwen2-v1.0", device: str = None, **kwargs
    ):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2,
            processor_class=ColQwen2Processor,
            device=device,
            **kwargs,
        )


class ColQwen2_5Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2.5 model."""

    def __init__(
        self, model_name: str = "vidore/colqwen2.5-v0.2", device: str = None, **kwargs
    ):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2_5,
            processor_class=ColQwen2_5Processor,
            device=device,
            **kwargs,
        )


colpali_training_datasets = {
    # TODO: Add the training datasets here
}

colqwen2 = ModelMeta(
    loader=partial(
        ColQwen2Wrapper,
        model_name="vidore/colqwen2-v1.0",
    ),
    name="vidore/colqwen2-v1.0",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-11-03",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=7200,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2-v1.0",
    similarity_fn_name="max_sim",
    use_instructions=False,
    training_datasets=colpali_training_datasets,
)

colqwen2_5 = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="vidore/colqwen2.5-v0.2",
    ),
    name="vidore/colqwen2.5-v0.2",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-01-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2.5-v0.2",
    similarity_fn_name="max_sim",
    use_instructions=False,
    training_datasets=colpali_training_datasets,
)

colnomic_7b = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="nomic-ai/colnomic-embed-multimodal-7b",
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="max_sim",
    use_instructions=False,
    training_datasets=colpali_training_datasets,
)

colnomic_3b = ModelMeta(
    loader=partial(
        ColQwen2_5Wrapper,
        model_name="nomic-ai/colnomic-embed-multimodal-3b",
    ),
    name="nomic-ai/colnomic-embed-multimodal-3b",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b",
    similarity_fn_name="max_sim",
    use_instructions=False,
    training_datasets=colpali_training_datasets,
)
