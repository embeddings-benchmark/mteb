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
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


def _row_id(row: dict[str, Any]) -> str:
    for key in ("id", "_id", "query-id", "query_id", "corpus-id", "corpus_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"MTEB row has no supported id field: {sorted(row)}")


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


class StructuralSeparatorSearch:
    """Exact single-vector retrieval with a learned structural separator.

    Documents insert the learned ``[unused2]`` token before a non-empty title
    and each punctuation-delimited sentence, then use the normalized CLS
    representation. Queries use the unchanged no-instruction BGE encoding, and
    matching is exact cosine search over the corpus or a supplied reranking
    candidate set.

    The motivation, algorithm, training procedure, BGE control results,
    falsification result, and limitations are documented at
    https://github.com/thu-nmrc/bge-small-structural-separator/blob/main/METHOD_CARD.md.
    """

    mteb_model_meta: ModelMeta | None = None
    max_length = 512
    separator_symbol = "[unused2]"
    separator_token_id = 3
    default_batch_size = 64
    embedding_dimension = 384

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        device: str | None = None,
        index_encoding_chunk_size: int = 4_096,
        query_search_batch_size: int = 64,
        document_search_block_size: int = 65_536,
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

        self.index_encoding_chunk_size = int(index_encoding_chunk_size)
        self.query_search_batch_size = int(query_search_batch_size)
        self.document_search_block_size = int(document_search_block_size)
        self.docids: list[str] = []
        self._docid_to_index: dict[str, int] = {}
        self.document_vectors = np.empty(
            (0, self.embedding_dimension), dtype=np.float32
        )
        self._index_path: Path | None = None

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        with torch.inference_mode():
            output = self.backbone(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_dict=True,
            ).last_hidden_state[:, 0]
            return F.normalize(output, dim=1).cpu().numpy()

    def _encode_documents(
        self, documents: list[dict[str, str]], *, batch_size: int
    ) -> np.ndarray:
        sequences = [
            _get_document_tokens(
                self.tokenizer,
                document,
                separator_token_id=self.separator_token_id,
                max_length=self.max_length,
            )
            for document in documents
        ]
        vectors = []
        for start in range(0, len(sequences), batch_size):
            input_ids, attention_mask = _pad(
                sequences[start : start + batch_size],
                int(self.tokenizer.pad_token_id),
            )
            vectors.append(self._encode(input_ids, attention_mask))
        return np.concatenate(vectors, axis=0)

    def _encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        vectors = []
        for start in range(0, len(queries), batch_size):
            tokens = self.tokenizer(
                queries[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            vectors.append(self._encode(tokens["input_ids"], tokens["attention_mask"]))
        return np.concatenate(vectors, axis=0)

    def close(self) -> None:
        empty = getattr(np, "empty", None)
        if empty is not None:
            self.document_vectors = empty(
                (0, self.embedding_dimension), dtype=np.float32
            )
        index_path = getattr(self, "_index_path", None)
        if index_path is not None:
            try:
                index_path.unlink()
            except FileNotFoundError:
                pass
            self._index_path = None

    def __del__(self) -> None:
        self.close()

    def _get_batch_size(self, encode_kwargs: EncodeKwargs) -> int:
        batch_size = int(encode_kwargs.get("batch_size", self.default_batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        batch_size = self._get_batch_size(encode_kwargs)
        self.close()
        self.docids = []
        self._docid_to_index = {}
        dimension = 0
        document_ids: list[str] = []
        documents: list[dict[str, str]] = []

        def flush(stream: Any) -> None:
            nonlocal dimension
            if not documents:
                return
            vectors = self._encode_documents(documents, batch_size=batch_size)
            dimension = int(vectors.shape[1])
            stream.write(vectors.tobytes(order="C"))
            self.docids.extend(document_ids)
            document_ids.clear()
            documents.clear()

        try:
            with tempfile.NamedTemporaryFile(
                prefix="structural-separator-index-", suffix=".f32", delete=False
            ) as stream:
                self._index_path = Path(stream.name)
                for row in corpus:
                    document_ids.append(_row_id(row))
                    documents.append(
                        {
                            "title": str(row.get("title", "") or ""),
                            "body": str(
                                row.get("body")
                                if row.get("body") is not None
                                else row.get("text", "")
                            ),
                        }
                    )
                    if len(documents) == self.index_encoding_chunk_size:
                        flush(stream)
                flush(stream)
        except Exception:
            self.close()
            raise
        if not self.docids:
            self.close()
            raise ValueError("Corpus must be nonempty")
        self._docid_to_index = {
            document_id: index for index, document_id in enumerate(self.docids)
        }
        self.document_vectors = np.memmap(
            self._index_path,
            mode="r",
            dtype=np.float32,
            shape=(len(self.docids), dimension),
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        batch_size = self._get_batch_size(encode_kwargs)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_map = {_row_id(row): str(row.get("text", "") or "") for row in queries}
        if not query_map:
            return {}
        query_ids = list(query_map)
        query_vectors = self._encode_queries(
            list(query_map.values()), batch_size=batch_size
        )
        if top_ranked is not None:
            return self._rerank_candidates(
                query_ids, query_vectors, top_ranked=top_ranked, top_k=top_k
            )

        output: RetrievalOutputType = {}
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

    def _rerank_candidates(
        self,
        query_ids: list[str],
        query_vectors: np.ndarray,
        *,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
    ) -> RetrievalOutputType:
        output: RetrievalOutputType = {}
        for query_id, query_vector in zip(query_ids, query_vectors):
            candidate_ids = list(top_ranked.get(query_id, []))
            missing = [
                document_id
                for document_id in candidate_ids
                if document_id not in self._docid_to_index
            ]
            if missing:
                raise KeyError(
                    f"Candidate documents are missing from the index: {missing[:3]}"
                )
            if not candidate_ids:
                output[query_id] = {}
                continue
            candidate_indices = [
                self._docid_to_index[document_id] for document_id in candidate_ids
            ]
            scores = query_vector @ self.document_vectors[candidate_indices].T
            ranked = sorted(
                zip(scores.tolist(), candidate_ids),
                key=lambda item: (-item[0], item[1]),
            )[:top_k]
            output[query_id] = {
                document_id: float(score) for score, document_id in ranked
            }
        return output


bge_small_structural_separator = ModelMeta(
    loader=StructuralSeparatorSearch,
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
