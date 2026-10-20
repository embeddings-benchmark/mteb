from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.cache_wrappers.cache_backends._ragged_storage import (
    _RaggedCacheBase,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.types import Array


class MultiVectorCache(_RaggedCacheBase):
    """Cache for multi-vector (late-interaction, e.g. ColBERT) embeddings.

    Each item maps to a variable number of token embeddings, stored contiguously in one
    `(total_tokens, dim)` float32 memmap.
    """

    _storage_file_name = "vectors.npy"
    _dtype = np.float32

    @property
    def _row_shape(self) -> tuple[int, ...] | None:
        return None if self.vector_dim is None else (self.vector_dim,)

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
        self._append_items(
            hashes,
            np.concatenate(new_vectors).reshape(-1, self.vector_dim),
            np.array([len(vector) for vector in new_vectors]),
        )

    def get_vector(self, item: dict[str, Any]) -> Array | None:
        """Retrieve the cached `(num_tokens, dim)` token embeddings."""
        return self._read(item)
