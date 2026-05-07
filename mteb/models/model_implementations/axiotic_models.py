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
        self.n_special_tokens = int(getattr(self.model.config, "n_special_tokens", 7))
        self.max_length = int(getattr(self.model.config, "max_seq_len", 1024))

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    @staticmethod
    def _task_for_prompt(prompt_type: PromptType | None) -> Any:
        if prompt_type == PromptType.query:
            return "QRY"
        if prompt_type == PromptType.document:
            return "DOC"
        return "SYM"

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        max_inner = self.max_length - 2
        encoded: list[list[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            shifted = [token_id + self.n_special_tokens for token_id in ids][:max_inner]
            encoded.append([2] + shifted + [3])

        max_len = max(len(ids) for ids in encoded)
        token_ids = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self.device
        )
        for row, ids in enumerate(encoded):
            tensor_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            token_ids[row, : len(ids)] = tensor_ids
            attention_mask[row, : len(ids)] = 1
        return token_ids, attention_mask

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
                token_ids, attention_mask = self._tokenize(texts)
                batch_embeddings = self.model.encode(
                    token_ids, attention_mask, task=task
                )
                embeddings.append(batch_embeddings.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)


ogma_micro = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-micro",
    model_type=["dense"],
    revision="d9d323709ab60f7833fc44dc2455a29b98ab324d",
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
    revision="300b6184ef8e53268171df68b10fd31d6cec1ea1",
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
    revision="9c3f997130f37ac632bf06408f7a93390a3dcf91",
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
    revision="7524c6e1b23ee1581748e6be39ab0ea91a336898",
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
