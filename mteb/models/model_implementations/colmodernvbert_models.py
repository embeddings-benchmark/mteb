from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

from .colpali_models import ColPaliEngineWrapper


class ColModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColModernVBERT models."""

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

    def get_text_embeddings(
        self,
        texts: DataLoader,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds = []
        with torch.no_grad():
            for batch in texts:
                inputs = self.processor.process_texts(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


class BiModernVBertWrapper(AbsEncoder):
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

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mdl = BiModernVBert.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.mdl.eval()

        self.processor = BiModernVBertProcessor.from_pretrained(model_name)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            return text_embeddings + image_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image inputs found")

    def get_text_embeddings(
        self,
        texts: DataLoader,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds = []
        with torch.no_grad():
            for batch in texts:
                inputs = self.processor.process_texts(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.mdl(**inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))
        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def get_image_embeddings(
        self,
        images: DataLoader,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds = []
        with torch.no_grad():
            for batch in images:
                inputs = self.processor.process_images(batch["image"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.mdl(**inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))
        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def similarity(self, a, b):
        return self.processor.score(a, b, device=self.device)


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
        trust_remote_code=True,
    ),
    name="ModernVBERT/colmodernvbert",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="e1e601df2542530091ade8a7b43c0bee99b58432",
    release_date="2025-10-02",
    modalities=["image", "text"],
    n_parameters=250_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=128,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ModernVBERT/colmodernvbert",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=COLMODERNVBERT_CITATION,
)


bimodernvbert = ModelMeta(
    loader=BiModernVBertWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ),
    name="ModernVBERT/bimodernvbert",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision=None,
    release_date="2025-10-02",
    modalities=["image", "text"],
    n_parameters=250_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=None,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ModernVBERT/bimodernvbert",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    citation=COLMODERNVBERT_CITATION,
)
