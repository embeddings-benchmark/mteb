from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


class OgmaWrapper:
    """MTEB wrapper for Axiotic Ogma embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        **_: Any,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True, use_fast=True
        )
        self.max_length = int(getattr(self.model.config, "max_seq_len", 1024))

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    @staticmethod
    def _task_for_prompt(prompt_type: PromptType | None) -> str:
        if prompt_type == PromptType.query:
            return "QRY"
        if prompt_type == PromptType.document:
            return "DOC"
        return "SYM"

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **_: Any,
    ) -> Array:
        task = self._task_for_prompt(prompt_type)
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in inputs:
                texts = [str(text) for text in batch["text"]]
                enc = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                batch_embeddings = self.model.encode(
                    enc["input_ids"], enc["attention_mask"], task=task
                )
                embeddings.append(batch_embeddings.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)


ogma_micro = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-micro",
    model_type=["dense"],
    revision="c9a793dacd593d1c0e336113ef1ac174a070217a",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=2_323_136,
    n_embedding_parameters=1_920_448,
    memory_usage_mb=8.9,
    max_tokens=1024,
    embed_dim=128,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-micro",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_mini = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-mini",
    model_type=["dense"],
    revision="580266301b651f100b19c928a65352e5fb57518a",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=3_511_744,
    n_embedding_parameters=1_920_448,
    memory_usage_mb=13.4,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-mini",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_small = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-small",
    model_type=["dense"],
    revision="761deba3f4d6f19cf799495417aeb7ee23abf7cd",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=8_596_352,
    n_embedding_parameters=3_840_896,
    memory_usage_mb=32.8,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-small",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_base = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-base",
    model_type=["dense"],
    revision="6c9cd11d41a04bae4b881c1f02cf6462511708b9",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=13_318_016,
    n_embedding_parameters=3_840_896,
    memory_usage_mb=50.8,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
