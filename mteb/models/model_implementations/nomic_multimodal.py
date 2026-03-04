from __future__ import annotations

import logging

import torch

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .colpali_models import ColPaliEngineWrapper
from .colqwen_models import COLNOMIC_LANGUAGES

CITATION = """
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal}
}"""

# https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source
TRAINING_DATA: set[str] = set()


logger = logging.getLogger(__name__)


class BiQwen2_5Wrapper(ColPaliEngineWrapper):  # noqa: N801
    """Wrapper for BiQwen2_5 dense (single-vector) embedding model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-multimodal-3b",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

        super().__init__(
            model_name=model_name,
            model_class=BiQwen2_5,
            processor_class=BiQwen2_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


nomic_embed_multimodal_3b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="nomic-ai/nomic-embed-multimodal-3b",
    model_type=["dense"],
    languages=COLNOMIC_LANGUAGES,
    revision="main",  # Will need to be updated with actual revision
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=6200,  # Estimated based on 3B vs 7B scaling
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=CITATION,
)

nomic_embed_multimodal_7b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="nomic-ai/nomic-embed-multimodal-7b",
    model_type=["dense"],
    languages=COLNOMIC_LANGUAGES,
    revision="1291f1b6ca07061b0329df9d5713c09b294be576",
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=CITATION,
)
