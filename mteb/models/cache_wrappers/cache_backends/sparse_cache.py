from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.cache_wrappers.cache_backends._ragged_storage import (
    _RaggedCacheBase,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.types import Array

_ENTRY_DTYPE = np.dtype([("index", np.int32), ("value", np.float32)])


class SparseCache(_RaggedCacheBase):
    """Cache for sparse embeddings (e.g. SPLADE), storing only the non-zero entries of each vector.

    Vectors added as sparse torch tensors are returned as 1D sparse COO tensors; dense vectors
    (as sparse encoders return for tasks that need dense embeddings) are returned as dense arrays.
    """

    _storage_file_name = "entries.npy"
    _dtype = _ENTRY_DTYPE

    def __init__(self, directory: str | Path, initial_rows: int = 100_000) -> None:
        super().__init__(directory, initial_rows=initial_rows)
        self.is_sparse: bool | None = None

    def _metadata(self) -> dict[str, Any]:
        return {**super()._metadata(), "is_sparse": self.is_sparse}

    def _load_metadata(self, metadata: dict[str, Any]) -> None:
        super()._load_metadata(metadata)
        self.is_sparse = metadata["is_sparse"]

    @staticmethod
    def _nonzero_entries(
        vectors: Array, positions: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the non-zero `(index, value)` entries of the requested rows, and their count per row."""
        import torch

        if isinstance(vectors, torch.Tensor) and vectors.is_sparse:
            coalesced = vectors.detach().cpu().coalesce()
            coordinates = coalesced.indices().numpy()
            all_values = coalesced.values().float().numpy()
            # Coalesced entries are sorted by row, so each row is a contiguous slice
            bounds = np.searchsorted(coordinates[0], np.arange(len(vectors) + 1))
            starts = bounds[positions]
            lengths = bounds[np.asarray(positions) + 1] - starts
            # Offset of each selected entry within its row, added to the row's start
            offsets = np.arange(lengths.sum()) - np.repeat(
                np.cumsum(lengths) - lengths, lengths
            )
            selected = np.repeat(starts, lengths) + offsets
            columns, values = coordinates[1, selected], all_values[selected]
        else:
            if isinstance(vectors, torch.Tensor):
                vectors = vectors.detach().cpu().float().numpy()
            dense = np.asarray(vectors)[positions]
            rows, columns = np.nonzero(dense)
            values = dense[rows, columns]
            lengths = np.bincount(rows, minlength=len(positions))

        entries = np.empty(len(columns), dtype=_ENTRY_DTYPE)
        entries["index"] = columns
        entries["value"] = values
        return entries, lengths

    def add(self, items: list[dict[str, Any]], vectors: Array) -> None:
        """Add the non-zero entries of each vector (a 2D sparse or dense array) to the cache."""
        import torch

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        is_sparse = isinstance(vectors, torch.Tensor) and vectors.is_sparse
        if self.vector_dim is None:
            self.vector_dim = int(vectors.shape[1])
            self.is_sparse = is_sparse
        elif self.is_sparse != is_sparse:
            raise ValueError(
                f"SparseCache in {self.directory} stores {'sparse' if self.is_sparse else 'dense'} "
                f"vectors, but {'sparse' if is_sparse else 'dense'} vectors were added."
            )
        elif vectors.shape[1] != self.vector_dim:
            raise ValueError(
                f"SparseCache in {self.directory} stores vectors of dimension {self.vector_dim}, "
                f"but vectors of dimension {vectors.shape[1]} were added."
            )

        positions, hashes = self._new_items(items)
        if positions:
            entries, lengths = self._nonzero_entries(vectors, positions)
            self._append_items(hashes, entries, lengths)

    def get_vector(self, item: dict[str, Any]) -> Array | None:
        """Retrieve the cached vector, as a sparse 1D tensor or a dense array."""
        entries = self._read(item)
        if entries is None or self.vector_dim is None:
            return None
        indices = entries["index"].astype(np.int64)
        values = np.ascontiguousarray(entries["value"])
        if self.is_sparse:
            import torch

            return torch.sparse_coo_tensor(
                torch.from_numpy(indices)[None, :],
                torch.from_numpy(values),
                (self.vector_dim,),
                is_coalesced=True,
            )
        dense = np.zeros(self.vector_dim, dtype=np.float32)
        dense[indices] = values
        return dense
