from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.cache_wrappers.cache_backends._ragged_storage import (
    _RaggedCacheBase,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mteb.types import Array


class MultiVectorCache(_RaggedCacheBase):
    """Cache for multi-vector (late-interaction, e.g. ColBERT) embeddings.

    Each item maps to a variable number of token embeddings, stored contiguously in one
    `(total_tokens, dim)` float32 memmap. Embeddings added as torch tensors are returned as
    tensors of the same dtype, so cached and newly encoded embeddings can be mixed.
    """

    _storage_file_name = "vectors.npy"
    _dtype = np.float32

    def __init__(self, directory: str | Path, initial_rows: int = 100_000) -> None:
        super().__init__(directory, initial_rows=initial_rows)
        # Name of the torch dtype of the added embeddings, or None if they were numpy arrays
        self.torch_dtype: str | None = None

    @property
    def _row_shape(self) -> tuple[int, ...] | None:
        return None if self.vector_dim is None else (self.vector_dim,)

    def _metadata(self) -> dict[str, Any]:
        return {**super()._metadata(), "torch_dtype": self.torch_dtype}

    def _load_metadata(self, metadata: dict[str, Any]) -> None:
        super()._load_metadata(metadata)
        self.torch_dtype = metadata["torch_dtype"]

    def add(
        self, items: list[dict[str, Any]], vectors: Array | Sequence[Array]
    ) -> None:
        """Add one `(num_tokens, dim)` array of token embeddings per item."""
        import torch

        positions, hashes = self._new_items(items)
        if not positions:
            return
        new_vectors = []
        for position in positions:
            vector = vectors[position]
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().float().numpy()
            new_vectors.append(np.asarray(vector, dtype=np.float32))
        if self.vector_dim is None:
            self.vector_dim = int(new_vectors[0].shape[-1])
            first = vectors[positions[0]]
            if isinstance(first, torch.Tensor):
                self.torch_dtype = str(first.dtype).removeprefix("torch.")
        self._append_items(
            hashes,
            np.concatenate(new_vectors).reshape(-1, self.vector_dim),
            np.array([len(vector) for vector in new_vectors]),
        )

    def get_vector(self, item: dict[str, Any]) -> Array | None:
        """Retrieve the cached `(num_tokens, dim)` token embeddings, in the type they were added as."""
        rows = self._read(item)
        if rows is None or self.torch_dtype is None:
            return rows
        import torch

        return torch.from_numpy(rows).to(getattr(torch, self.torch_dtype))
