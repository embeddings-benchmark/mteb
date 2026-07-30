from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from mteb.models import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _get_document_tokens(
    tokenizer: Any,
    raw: dict[str, str],
    *,
    separator_token_id: int,
    max_length: int,
) -> list[int]:
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer must define CLS and SEP token ids")
    ids = [int(tokenizer.cls_token_id)]
    title = str(raw.get("title", "") or "").strip()
    if title:
        ids.append(separator_token_id)
        ids.extend(tokenizer.encode(title, add_special_tokens=False))
    body = raw.get("body")
    if body is None:
        body = raw.get("text", "")
    for sentence in _split_sentences(str(body or "")):
        ids.append(separator_token_id)
        ids.extend(tokenizer.encode(sentence, add_special_tokens=False))
    if len(ids) == 1:
        ids.append(separator_token_id)
    return ids[: max_length - 1] + [int(tokenizer.sep_token_id)]


def _pad(
    sequences: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, sequences))
    input_ids = torch.full((len(sequences), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


class StructuralSeparatorEncoder:
    """Single-vector encoder using a learned document-structure separator.

    Documents insert the learned ``[unused2]`` token before a non-empty title
    and each punctuation-delimited sentence. Queries and non-document inputs
    use the unchanged no-instruction BGE encoding. All inputs use the normalized
    CLS representation and cosine matching.

    The motivation, algorithm, training procedure, BGE control results,
    falsification result, and limitations are documented at
    https://github.com/thu-nmrc/bge-small-structural-separator/blob/main/METHOD_CARD.md.
    """

    mteb_model_meta: ModelMeta | None = None
    max_length = 512
    separator_symbol = "[unused2]"
    separator_token_id = 3
    embedding_dimension = 384

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.backbone = (
            AutoModel.from_pretrained(model_name, revision=revision)
            .to(self.device)
            .eval()
        )
        if (
            self.tokenizer.convert_ids_to_tokens(self.separator_token_id)
            != self.separator_symbol
        ):
            raise RuntimeError("Separator token does not match the pinned tokenizer")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token id")

    def _encode_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        with torch.inference_mode():
            output = self.backbone(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_dict=True,
            ).last_hidden_state[:, 0]
            return F.normalize(output, dim=1).cpu().numpy()

    def _encode_documents(self, batch: BatchedInput) -> np.ndarray:
        texts = list(batch["text"])
        bodies = list(batch.get("body", texts))
        titles = list(batch.get("title", [""] * len(bodies)))
        sequences = [
            _get_document_tokens(
                self.tokenizer,
                {
                    "title": str(title or ""),
                    "body": str(body or ""),
                },
                separator_token_id=self.separator_token_id,
                max_length=self.max_length,
            )
            for title, body in zip(titles, bodies, strict=True)
        ]
        input_ids, attention_mask = _pad(
            sequences,
            int(self.tokenizer.pad_token_id),
        )
        return self._encode_batch(input_ids, attention_mask)

    def _encode_texts(self, batch: BatchedInput) -> np.ndarray:
        tokens = self.tokenizer(
            list(batch["text"]),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return self._encode_batch(tokens["input_ids"], tokens["attention_mask"])

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
        vectors = []
        for batch in inputs:
            if prompt_type == PromptType.document:
                vectors.append(self._encode_documents(batch))
            else:
                vectors.append(self._encode_texts(batch))
        if not vectors:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    @staticmethod
    def similarity(embeddings1: Array, embeddings2: Array) -> Array:
        if isinstance(embeddings1, torch.Tensor) or isinstance(
            embeddings2, torch.Tensor
        ):
            tensor1 = torch.as_tensor(embeddings1)
            tensor2 = torch.as_tensor(embeddings2)
            return tensor1 @ tensor2.T
        return np.asarray(embeddings1) @ np.asarray(embeddings2).T

    @staticmethod
    def similarity_pairwise(embeddings1: Array, embeddings2: Array) -> Array:
        if isinstance(embeddings1, torch.Tensor) or isinstance(
            embeddings2, torch.Tensor
        ):
            tensor1 = torch.as_tensor(embeddings1)
            tensor2 = torch.as_tensor(embeddings2)
            return (tensor1 * tensor2).sum(dim=1)
        return (np.asarray(embeddings1) * np.asarray(embeddings2)).sum(axis=1)


bge_small_structural_separator = ModelMeta(
    loader=StructuralSeparatorEncoder,
    name="thu-nmrc/bge-small-structural-separator",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="9a0a8aa92400202dd1ef6950ed9cd4a116dfb03d",
    release_date="2026-07-14",
    n_parameters=33_360_000,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=None,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://github.com/thu-nmrc/bge-small-structural-separator/blob/main/METHOD_CARD.md",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    public_training_code="https://github.com/thu-nmrc/bge-small-structural-separator/tree/e8b0a9325409d791981b7410679ae8c152fd6e00/training",
    public_training_data="https://allenai.org/data/s2orc",
    training_datasets=set(),
    adapted_from="BAAI/bge-small-en-v1.5",
    citation=None,
    contacts=["thu-nmrc"],
)
