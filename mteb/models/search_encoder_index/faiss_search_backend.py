import logging
from collections.abc import Callable

import faiss
import numpy as np
import torch
from faiss import IndexFlatIP, IndexFlatL2

from mteb import EncoderProtocol
from mteb.models.model_meta import ScoringFunction
from mteb.types import Array, TopRankedDocumentsType

logger = logging.getLogger(__name__)


class FaissEncoderSearchBackend:
    """FAISS-based backend for encoder-based search.

    Supports both full-corpus retrieval and reranking (via `top_ranked`).

    Notes:
        - Stores *all* embeddings in memory (IndexFlatIP or IndexFlatL2).
        - Expects embeddings to be normalized if cosine similarity is desired.
    """

    _normalize: bool = False

    def __init__(self, model: EncoderProtocol) -> None:
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        if (
            model.mteb_model_meta.similarity_fn_name == "dot"
            or model.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT
        ):
            self.index_type = IndexFlatL2
        else:
            self.index_type = IndexFlatIP
            self._normalize = True

        self.idxs: list[str] = []
        self.index: faiss.Index | None = None

    def add_document(self, embeddings: Array, idxs: list[str]) -> None:
        """Add all document embeddings and their IDs to FAISS index."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = embeddings.astype(np.float32)
        self.idxs.extend(idxs)

        if self._normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = self.index_type(dim)

        self.index.add(embeddings)
        logger.info(f"FAISS index built with {len(idxs)} vectors of dim {dim}.")

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search using FAISS."""
        if self.index is None:
            raise ValueError("No index built. Call add_document() first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if self._normalize:
            faiss.normalize_L2(embeddings)

        if top_ranked is not None:
            if query_idx_to_id is None:
                raise ValueError("query_idx_to_id must be provided when reranking.")

            doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.idxs)}
            scores_all: list[list[float]] = []
            idxs_all: list[list[int]] = []

            for query_idx, query_emb in enumerate(embeddings):
                query_id = query_idx_to_id[query_idx]
                ranked_ids = top_ranked.get(query_id)
                if not ranked_ids:
                    logger.warning(f"No top-ranked documents for query {query_id}")
                    scores_all.append([])
                    idxs_all.append([])
                    continue

                candidate_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
                d = self.index.d
                candidate_embs = np.zeros((len(candidate_indices), d), dtype=np.float32)
                for j, idx in enumerate(candidate_indices):
                    candidate_embs[j] = self.index.reconstruct(idx)

                scores = similarity_fn(
                    torch.as_tensor(query_emb).unsqueeze(0),
                    torch.as_tensor(candidate_embs),
                )

                values, indices = torch.topk(
                    torch.as_tensor(scores),
                    k=min(top_k, len(candidate_indices)),
                    dim=1,
                    largest=True,
                )
                scores_all.append(values.squeeze(0).cpu().tolist())
                idxs_all.append(indices.squeeze(0).cpu().tolist())

            return scores_all, idxs_all

        documents, ids = self.index.search(embeddings.astype(np.float32), top_k)
        return documents.tolist(), ids.tolist()

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
        self.index = None
        self.idxs = []
