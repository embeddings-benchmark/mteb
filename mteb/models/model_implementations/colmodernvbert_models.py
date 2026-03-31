from __future__ import annotations

from typing import Any

import torch

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .colpali_models import ColPaliEngineWrapper, COLPALI_TRAINING_DATA

class ColModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColModernVBert model."""

    def __init__(
        self,
        model_name: str = "ModernVBERT/colmodernvbert",
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColModernVBert,
            processor_class=ColModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])

class BiModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for BiModernVBERT models."""

    def __init__(
        self,
        model_name: str = "ModernVBERT/bimodernvbert",
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=BiModernVBert,
            processor_class=BiModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])

COLMODERNVBERT_CITATION = """
@misc{teiletche2025modernvbertsmallervisualdocument,
  title={ModernVBERT: Towards Smaller Visual Document Retrievers},
  author={Paul Teiletche and Quentin Macé and Max Conti and Antonio Loison and Gautier Viaud and Pierre Colombo and Manuel Faysse},
  year={2025},
  eprint={2510.01149},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2510.01149}
}
"""


colmodernvbert = ModelMeta(
    loader=ColModernVBertWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float32,
    ),
    name="ModernVBERT/colmodernvbert",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="9052fc0ef8acf9fb764681ce0315a7f89ea7d276",
    release_date="2025-10-01",
    modalities=["image", "text"],
    n_parameters=252_002_304,
    n_embedding_parameters=38_713_344,
    memory_usage_mb=961,
    max_tokens=8192,
    embed_dim=128,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/modernvbert",
    public_training_data="https://huggingface.co/collections/ModernVBERT/colmodernvbert",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ModernVBERT/colmodernvbert",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLMODERNVBERT_CITATION,
)


bimodernvbert = ModelMeta(
    loader=BiModernVBertWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float32,
    ),
    name="ModernVBERT/bimodernvbert",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="09ed8566839b68207c76e4869f1bb4406edc2fcd",
    release_date="2025-10-01",
    modalities=["image", "text"],
    n_parameters=252_002_304,
    n_embedding_parameters=38_713_344,
    memory_usage_mb=961,
    max_tokens=8192,
    embed_dim=None,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/modernvbert",
    public_training_data="https://huggingface.co/collections/ModernVBERT/colmodernvbert",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ModernVBERT/bimodernvbert",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLMODERNVBERT_CITATION,
)
