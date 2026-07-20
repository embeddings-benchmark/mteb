from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from mteb.models import ModelMeta

if TYPE_CHECKING:
    from collections.abc import Iterable

MODEL_NAME = "thu-nmrc/bge-small-structural-separator"
MODEL_REVISION = "9a0a8aa92400202dd1ef6950ed9cd4a116dfb03d"
BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BASE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
REPOSITORY_URL = "https://huggingface.co/thu-nmrc/bge-small-structural-separator"
TRAINING_REPOSITORY_URL = (
    "https://github.com/thu-nmrc/bge-small-structural-separator"
)
TRAINING_REPOSITORY_REVISION = "e8b0a9325409d791981b7410679ae8c152fd6e00"
SEPARATOR_SYMBOL = "[unused2]"
SEPARATOR_TOKEN_ID = 3
MAX_LENGTH = 512


def _row_id(row: dict[str, Any]) -> str:
    for key in ("id", "_id", "query-id", "query_id", "corpus-id", "corpus_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"MTEB row has no supported id field: {sorted(row)}")


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _document_ids(
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
    for sentence in _split_sentences(str(raw.get("text", "") or "")):
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


class StructuralSeparatorSearch:
    """One-vector exact retrieval with a learned shared structural separator."""

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        device: str | None = None,
        model_source: str | Path | None = None,
        batch_size: int = 64,
        index_encoding_chunk_size: int = 4_096,
        query_search_batch_size: int = 64,
        document_search_block_size: int = 65_536,
        **_: Any,
    ) -> None:
        if model_name != MODEL_NAME or revision not in {None, MODEL_REVISION}:
            raise ValueError(f"Unsupported model identity: {model_name}@{revision}")

        from transformers import AutoModel, AutoTokenizer

        source = str(model_source or MODEL_NAME)
        source_revision = None if model_source is not None else MODEL_REVISION
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=source_revision)
        self.backbone = (
            AutoModel.from_pretrained(source, revision=source_revision)
            .to(self.device)
            .eval()
        )
        self.separator_token_id = SEPARATOR_TOKEN_ID
        if self.tokenizer.convert_ids_to_tokens(SEPARATOR_TOKEN_ID) != SEPARATOR_SYMBOL:
            raise RuntimeError("Separator token does not match the pinned tokenizer")

        self.max_length = MAX_LENGTH
        self.batch_size = int(batch_size)
        self.index_encoding_chunk_size = int(index_encoding_chunk_size)
        self.query_search_batch_size = int(query_search_batch_size)
        self.document_search_block_size = int(document_search_block_size)
        self.docids: list[str] = []
        self.document_vectors = np.empty((0, 384), dtype=np.float32)
        self._index_path: Path | None = None
        self._model_meta: ModelMeta | None = None

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._model_meta or bge_small_structural_separator

    @mteb_model_meta.setter
    def mteb_model_meta(self, value: ModelMeta) -> None:
        self._model_meta = value

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        with torch.inference_mode():
            output = self.backbone(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_dict=True,
            ).last_hidden_state[:, 0]
            return (
                F.normalize(output, dim=1).cpu().numpy().astype(np.float32, copy=False)
            )

    def _encode_documents(self, documents: list[dict[str, str]]) -> np.ndarray:
        sequences = [
            _document_ids(
                self.tokenizer,
                document,
                separator_token_id=self.separator_token_id,
                max_length=self.max_length,
            )
            for document in documents
        ]
        vectors = []
        for start in range(0, len(sequences), self.batch_size):
            input_ids, attention_mask = _pad(
                sequences[start : start + self.batch_size],
                int(self.tokenizer.pad_token_id),
            )
            vectors.append(self._encode(input_ids, attention_mask))
        return np.concatenate(vectors, axis=0)

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(queries), self.batch_size):
            tokens = self.tokenizer(
                queries[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            vectors.append(self._encode(tokens["input_ids"], tokens["attention_mask"]))
        return np.concatenate(vectors, axis=0)

    def close(self) -> None:
        numpy = globals().get("np")
        if numpy is not None:
            self.document_vectors = numpy.empty((0, 384), dtype=numpy.float32)
        index_path = getattr(self, "_index_path", None)
        if index_path is not None:
            try:
                index_path.unlink()
            except FileNotFoundError:
                pass
            self._index_path = None

    def __del__(self) -> None:
        self.close()

    def index(
        self,
        corpus: Iterable[dict[str, Any]],
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        num_proc: int | None,
    ) -> None:
        del task_metadata, hf_split, hf_subset, encode_kwargs, num_proc
        self.close()
        self.docids = []
        handle = tempfile.NamedTemporaryFile(
            prefix="structural-separator-index-", suffix=".f32", delete=False
        )
        handle.close()
        self._index_path = Path(handle.name)
        dimension = 0
        document_ids: list[str] = []
        documents: list[dict[str, str]] = []

        def flush(stream: Any) -> None:
            nonlocal dimension
            if not documents:
                return
            vectors = self._encode_documents(documents)
            dimension = int(vectors.shape[1])
            stream.write(vectors.tobytes(order="C"))
            self.docids.extend(document_ids)
            document_ids.clear()
            documents.clear()

        with self._index_path.open("wb") as stream:
            for row in corpus:
                document_ids.append(_row_id(row))
                documents.append(
                    {
                        "title": str(row.get("title", "") or ""),
                        "text": str(row.get("text", "") or ""),
                    }
                )
                if len(documents) == self.index_encoding_chunk_size:
                    flush(stream)
            flush(stream)
        if not self.docids:
            self.close()
            raise ValueError("Corpus must be nonempty")
        self.document_vectors = np.memmap(
            self._index_path,
            mode="r",
            dtype=np.float32,
            shape=(len(self.docids), dimension),
        )

    def search(
        self,
        queries: Iterable[dict[str, Any]],
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: dict[str, list[str]] | None = None,
        num_proc: int | None,
    ) -> dict[str, dict[str, float]]:
        del task_metadata, hf_split, hf_subset, encode_kwargs, num_proc
        if top_ranked is not None:
            raise ValueError(
                "This is a first-stage retriever; reranking inputs are not accepted"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_map = {_row_id(row): str(row.get("text", "") or "") for row in queries}
        if not query_map:
            return {}
        query_ids = list(query_map)
        query_vectors = self._encode_queries(list(query_map.values()))
        output: dict[str, dict[str, float]] = {}
        for query_start in range(0, len(query_ids), self.query_search_batch_size):
            query_stop = min(query_start + self.query_search_batch_size, len(query_ids))
            retained_by_query: list[list[tuple[float, str]]] = [
                [] for _ in range(query_stop - query_start)
            ]
            for document_start in range(
                0, len(self.docids), self.document_search_block_size
            ):
                document_stop = min(
                    document_start + self.document_search_block_size, len(self.docids)
                )
                scores = (
                    query_vectors[query_start:query_stop]
                    @ self.document_vectors[document_start:document_stop].T
                )
                for row, retained in enumerate(retained_by_query):
                    keep = min(top_k, document_stop - document_start)
                    if keep < document_stop - document_start:
                        threshold = float(np.partition(scores[row], -keep)[-keep])
                        local = np.flatnonzero(scores[row] > threshold).tolist()
                        tied = np.flatnonzero(scores[row] == threshold).tolist()
                        tied.sort(key=lambda index: self.docids[document_start + index])
                        local.extend(tied[: keep - len(local)])
                    else:
                        local = list(range(document_stop - document_start))
                    retained.extend(
                        (float(scores[row, index]), self.docids[document_start + index])
                        for index in local
                    )
                    retained.sort(key=lambda item: (-item[0], item[1]))
                    del retained[top_k:]
            for query_id, retained in zip(
                query_ids[query_start:query_stop], retained_by_query
            ):
                output[query_id] = {
                    document_id: score for score, document_id in retained
                }
        return output


bge_small_structural_separator = ModelMeta(
    loader=StructuralSeparatorSearch,
    name=MODEL_NAME,
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision=MODEL_REVISION,
    release_date="2026-07-14",
    n_parameters=33_360_000,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=None,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference=REPOSITORY_URL,
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    public_training_code=(
        f"{TRAINING_REPOSITORY_URL}/tree/{TRAINING_REPOSITORY_REVISION}/training"
    ),
    public_training_data="https://allenai.org/data/s2orc",
    training_datasets=set(),
    adapted_from=BASE_MODEL_NAME,
    citation=None,
    contacts=["thu-nmrc"],
)
