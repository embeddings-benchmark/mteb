import logging

import torch

from mteb._requires_package import (
    requires_package,
)
from mteb.models.model_meta import ModelMeta

from .colpali_models import (
    COLPALI_CITATION,
    COLPALI_TRAINING_DATA,
    ColPaliEngineWrapper,
)

logger = logging.getLogger(__name__)


class ColSmolWrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else "default"
            )

        super().__init__(
            model_name=model_name,
            model_class=ColIdefics3,
            processor_class=ColIdefics3Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


colsmol_256m = ModelMeta(
    loader=ColSmolWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colSmol-256M",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-01-22",
    modalities=["image", "text"],
    n_parameters=256_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colSmol-256M",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

colsmol_500m = ModelMeta(
    loader=ColSmolWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="vidore/colSmol-500M",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="1aa9325cba7ed2b3b9b97ede4d55026322504902",
    release_date="2025-01-22",
    modalities=["image", "text"],
    n_parameters=500_000_000,
    memory_usage_mb=1200,
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colSmol-500M",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)
