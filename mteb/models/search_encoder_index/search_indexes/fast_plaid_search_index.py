from __future__ import annotations

import logging
import shutil
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mteb._requires_package import _is_package_available
from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from fast_plaid import search as fast_plaid_search

    from mteb.models.model_meta import MODEL_TYPES
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, TopRankedDocumentsType


logger = logging.getLogger(__name__)


def _as_token_tensors(embeddings: Array | list[Array]) -> list[torch.Tensor]:
    """Convert multi-vector embeddings to a list of float32 CPU `(n_tokens, dim)` tensors."""
    import torch

    return [torch.as_tensor(e).detach().to("cpu", torch.float32) for e in embeddings]


class FastPlaidSearchIndex:
    """[PLAID](https://arxiv.org/abs/2205.09707) backend for multi-vector (late interaction) search, based on [fast-plaid](https://github.com/lightonai/fast-plaid).

    Supports both full-corpus retrieval and reranking (via `top_ranked`).

    Notes:
        - Only MaxSim models are supported.
        - Search is approximate: documents are retrieved through a centroid index and
          the best `n_full_scores` candidates are scored with MaxSim on the compressed
          (`nbits`) residuals. This also holds for reranking, where PLAID only restricts the
          search to each query's candidates.
        - `similarity_fn` passed to `search` is ignored, PLAID always scores with MaxSim.
        - Because results are approximate, the index parameters are saved as `experiment_kwargs`
          of the model when used through `SearchEncoderWrapper` (see `experiment_kwargs`).
    """

    supported_model_types: ClassVar[tuple[MODEL_TYPES, ...]] = ("late-interaction",)

    def __init__(
        self,
        model: EncoderProtocol,
        *,
        index_dir: str | Path | None = None,
        device: str | list[str] | None = None,
        nbits: int = 4,
        kmeans_niters: int = 4,
        n_ivf_probe: int = 8,
        n_full_scores: int = 4096,
        seed: int = 42,
    ) -> None:
        """Create the backend.

        Args:
            model: Model the index is built for. Must use MaxSim as similarity function.
            index_dir: Directory to store the index in. A temporary directory is created when
                not given; it is emptied on `clear` and removed with the backend.
            device: Device(s) used by fast-plaid, e.g. `"cpu"` or `"cuda:0"`. Auto-detected when not given.
            nbits: Number of bits used to quantize residuals.
            kmeans_niters: Number of K-means iterations when building centroids.
            n_ivf_probe: Number of clusters probed per query token.
            n_full_scores: Number of candidate documents scored exactly per query.
            seed: Random seed for K-means.
        """
        if not _is_package_available("fast_plaid"):
            raise ImportError(
                "fast-plaid is required for FastPlaidSearchIndex. Please install with `pip install mteb[fast-plaid]`."
            )

        similarity_fn_name = model.mteb_model_meta.similarity_fn_name
        if similarity_fn_name is not ScoringFunction.MAX_SIM:
            raise ValueError(
                f"FastPlaidSearchIndex only supports {ScoringFunction.MAX_SIM} models, "
                f"got {similarity_fn_name}."
            )

        # `FastPlaid.create` cleans the directory, so one directory is reused for every task
        self._owns_index_dir = index_dir is None
        self.index_dir = (
            Path(tempfile.mkdtemp(prefix="mteb-fast-plaid-"))
            if index_dir is None
            else Path(index_dir)
        )
        if self._owns_index_dir:
            weakref.finalize(self, shutil.rmtree, self.index_dir, ignore_errors=True)
        self.device = device
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.n_ivf_probe = n_ivf_probe
        self.n_full_scores = n_full_scores
        self.seed = seed

        self.idxs: list[str] = []
        self.index: fast_plaid_search.FastPlaid | None = None

    @property
    def experiment_kwargs(self) -> dict[str, Any]:
        """Parameters that change the (approximate) search results.

        `SearchEncoderWrapper` adds them to the model's `experiment_kwargs`, so PLAID results are
        saved as an experiment, separate from exact MaxSim results. `index_dir` and `device` are
        left out: they only change where the index lives and how fast it is searched.
        """
        return {
            "index_backend": "fast_plaid",
            "plaid_nbits": self.nbits,
            "plaid_kmeans_niters": self.kmeans_niters,
            "plaid_n_ivf_probe": self.n_ivf_probe,
            "plaid_n_full_scores": self.n_full_scores,
            "plaid_seed": self.seed,
        }

    def add_documents(self, embeddings: Array, idxs: list[str]) -> None:
        """Add document multi-vector embeddings and their IDs to the PLAID index."""
        from fast_plaid import search as fast_plaid_search

        documents = _as_token_tensors(embeddings)
        if self.index is None:
            self.index = fast_plaid_search.FastPlaid(
                index=str(self.index_dir), device=self.device
            )
            self.index.create(
                documents_embeddings=documents,
                kmeans_niters=self.kmeans_niters,
                nbits=self.nbits,
                seed=self.seed,
            )
        else:
            self.index.update(documents_embeddings=documents)

        self.idxs.extend(idxs)
        logger.info(f"PLAID index built with {len(idxs)} documents.")

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search using PLAID.

        Returns:
            For each query, the top-k scores and indices. Indices point into the added corpus,
            or, when reranking, into the query's `top_ranked` list.
        """
        if self.index is None:
            raise ValueError("No index built. Call add_documents() first.")

        queries = _as_token_tensors(embeddings)
        candidates: list[list[int]] = []
        if top_ranked is not None:
            if query_idx_to_id is None:
                raise ValueError("query_idx_to_id must be provided when reranking.")
            doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.idxs)}
            for query_idx in range(len(queries)):
                query_id = query_idx_to_id[query_idx]
                ranked_ids = top_ranked.get(query_id)
                if not ranked_ids:
                    msg = f"No top-ranked documents for query {query_id}"
                    logger.warning(msg)
                    warnings.warn(msg, stacklevel=2)
                    candidates.append([])
                    continue
                candidates.append([doc_id_to_idx[doc_id] for doc_id in ranked_ids])

        scores_all: list[list[float]] = [[] for _ in queries]
        idxs_all: list[list[int]] = [[] for _ in queries]
        if top_ranked is None:
            results = self.index.search(
                queries_embeddings=queries,
                top_k=min(top_k, len(self.idxs)),
                n_ivf_probe=self.n_ivf_probe,
                n_full_scores=self.n_full_scores,
                show_progress=False,
            )
            for query_idx, query_results in enumerate(results):
                idxs_all[query_idx] = [doc_idx for doc_idx, _ in query_results]
                scores_all[query_idx] = [float(score) for _, score in query_results]
            return scores_all, idxs_all

        # fast-plaid takes one subset per query, so queries without candidates are left out
        active = [i for i, cand in enumerate(candidates) if cand]
        if not active:
            return scores_all, idxs_all
        subset = [candidates[i] for i in active]
        results = self.index.search(
            queries_embeddings=[queries[i] for i in active],
            top_k=min(top_k, max(len(cand) for cand in subset)),
            n_ivf_probe=self.n_ivf_probe,
            n_full_scores=self.n_full_scores,
            show_progress=False,
            subset=subset,
        )
        for query_idx, query_results in zip(active, results, strict=True):
            position = {
                doc_idx: pos for pos, doc_idx in enumerate(candidates[query_idx])
            }
            idxs_all[query_idx] = [position[doc_idx] for doc_idx, _ in query_results]
            scores_all[query_idx] = [float(score) for _, score in query_results]
        return scores_all, idxs_all

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
        self.index = None
        self.idxs = []
        # Free the disk space of the temporary index, but keep the directory for the next task
        if self._owns_index_dir:
            shutil.rmtree(self.index_dir, ignore_errors=True)
            self.index_dir.mkdir()
