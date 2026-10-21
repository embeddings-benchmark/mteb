from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mteb._content_hashes import hash_item

if TYPE_CHECKING:
    from mteb.types import Array

logger = logging.getLogger(__name__)


class _GrowableMemmap:
    """Append-only memory-mapped array that doubles its capacity when full.

    Rows have a fixed `row_shape` (e.g. `()` for scalars or `(dim,)` for vectors), so variable-length
    items are stored as contiguous `[start, end)` row ranges. If `row_shape` is None, it is taken
    from the first `append`.
    """

    def __init__(
        self,
        path: Path,
        dtype: np.typing.DTypeLike,
        row_shape: tuple[int, ...] | None,
        initial_rows: int,
        used: int,
    ) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.row_shape = row_shape
        self.initial_rows = initial_rows
        self.used = used
        self.data: np.memmap[Any, np.dtype[Any]] | None = None
        if self.path.exists() and self.row_shape is not None:
            row_bytes = self.dtype.itemsize * int(np.prod(self.row_shape))
            self.data = self._open("r+", self.path.stat().st_size // row_bytes)

    def _open(
        self, mode: Literal["r+", "w+"], n_rows: int
    ) -> np.memmap[Any, np.dtype[Any]]:
        if self.row_shape is None:
            raise RuntimeError(f"Row shape of {self.path} is not set")
        return np.memmap(
            self.path, dtype=self.dtype, mode=mode, shape=(n_rows, *self.row_shape)
        )

    def append(self, rows: np.ndarray) -> int:
        """Append `rows` and return the index of the first one."""
        if self.row_shape is None:
            self.row_shape = rows.shape[1:]
        start = self.used
        end = start + len(rows)
        if self.data is None:
            self.data = self._open("w+", max(self.initial_rows, end))
        elif end > len(self.data):
            capacity = len(self.data)
            while capacity < end:
                capacity *= 2
            logger.info(f"Growing {self.path} from {len(self.data)} to {capacity} rows")
            self.data.flush()
            # `r+` with a larger shape extends the file in place
            self.data = self._open("r+", capacity)
        self.data[start:end] = rows
        self.used = end
        return start

    def read(self, start: int, end: int) -> np.ndarray:
        """Return a copy of rows `[start, end)`."""
        if self.data is None:
            return np.empty((0, *(self.row_shape or ())), dtype=self.dtype)
        return np.array(self.data[start:end])

    def flush(self) -> None:
        if self.data is not None:
            self.data.flush()


class _RaggedCacheBase:
    """Index, metadata and storage handling for caches storing variable-length items.

    Each item is identified by its content hash and maps to a `[start, end)` range of rows in a
    single `_GrowableMemmap`. The index is an append-only JSON-lines file, so saving only writes
    the items added since the last save.
    """

    _storage_file_name: str
    _dtype: np.typing.DTypeLike

    def __init__(self, directory: str | Path, initial_rows: int = 100_000) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.directory / "index.jsonl"
        self.metadata_file = self.directory / "metadata.json"
        self.initial_rows = initial_rows
        self.hash_to_range: dict[str, tuple[int, int]] = {}
        self.vector_dim: int | None = None
        self._unsaved: list[str] = []
        self._storage: _GrowableMemmap | None = None
        logger.info(f"Initialized {type(self).__name__} in {self.directory}")

    @property
    def _row_shape(self) -> tuple[int, ...] | None:
        """Shape of one storage row, or None if it is only known after the first `add`."""
        return ()

    def _metadata(self) -> dict[str, Any]:
        return {"dim": self.vector_dim}

    def _load_metadata(self, metadata: dict[str, Any]) -> None:
        self.vector_dim = metadata["dim"]

    @property
    def storage(self) -> _GrowableMemmap:
        if self._storage is None:
            self._storage = _GrowableMemmap(
                self.directory / self._storage_file_name,
                self._dtype,
                row_shape=self._row_shape,
                initial_rows=self.initial_rows,
                used=max((end for _, end in self.hash_to_range.values()), default=0),
            )
        return self._storage

    def _append_items(
        self, item_hashes: list[str], rows: np.ndarray, lengths: Array
    ) -> None:
        """Store `rows`, of which consecutive runs of `lengths` belong to each item."""
        start = self.storage.append(rows)
        ends = start + np.cumsum(lengths)
        for item_hash, item_end, length in zip(
            item_hashes, ends.tolist(), lengths, strict=True
        ):
            self.hash_to_range[item_hash] = (item_end - int(length), item_end)
        self._unsaved.extend(item_hashes)

    def save(self) -> None:
        """Persist the vectors, metadata and newly added index entries to disk."""
        self.storage.flush()
        tmp_metadata_file = self.metadata_file.with_suffix(".json.tmp")
        with tmp_metadata_file.open("w", encoding="utf-8") as f:
            json.dump(self._metadata(), f)
        tmp_metadata_file.replace(self.metadata_file)
        with self.index_file.open("a", encoding="utf-8") as f:
            for item_hash in self._unsaved:
                start, end = self.hash_to_range[item_hash]
                f.write(json.dumps([item_hash, start, end]) + "\n")
        self._unsaved = []
        logger.info(f"Saved {type(self).__name__} to {self.directory}")

    def load(self) -> None:
        """Load the index and metadata from disk."""
        self.hash_to_range = {}
        self._unsaved = []
        self._storage = None
        if not (self.index_file.exists() and self.metadata_file.exists()):
            return
        with self.metadata_file.open(encoding="utf-8") as f:
            self._load_metadata(json.load(f))
        with self.index_file.open(encoding="utf-8") as f:
            for line in f:
                item_hash, start, end = json.loads(line)
                self.hash_to_range[item_hash] = (start, end)
        logger.info(
            f"Loaded {type(self).__name__} with {len(self.hash_to_range)} items from {self.directory}"
        )

    def _new_items(self, items: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
        """Return the positions and hashes of items not cached yet, skipping duplicates."""
        positions: list[int] = []
        hashes: list[str] = []
        seen: set[str] = set()
        for position, item in enumerate(items):
            item_hash = hash_item(item)
            if item_hash in self.hash_to_range or item_hash in seen:
                continue
            seen.add(item_hash)
            positions.append(position)
            hashes.append(item_hash)
        return positions, hashes

    def _read(self, item: dict[str, Any]) -> np.ndarray | None:
        """Return the stored rows of `item`, or None if it isn't cached."""
        range_ = self.hash_to_range.get(hash_item(item))
        if range_ is None:
            return None
        return self.storage.read(*range_)

    def __contains__(self, item: dict[str, Any]) -> bool:
        return hash_item(item) in self.hash_to_range

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Flush and release the storage file."""
        if getattr(self, "_storage", None) is not None:
            self.storage.flush()
        self._storage = None
