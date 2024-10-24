from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.models.wrapper import Wrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextVectorMap:
    def __init__(
        self,
        directory: Union[str | Path],
        initial_vectors: int = 100000,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.directory / "vectors.npy"
        self.index_file = self.directory / "index.json"
        self.dimension_file = self.directory / "dimension"
        self.hash_to_index: Dict[str, int] = {}
        self.vectors: Optional[np.memmap] = None
        self.vector_dim: Optional[int] = None
        self.initial_vectors = initial_vectors
        logger.info(f"Initialized TextVectorMap in directory: {self.directory}")
        self._initialize_vectors_file()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def add(self, text: str, vector: np.ndarray) -> None:
        try:
            if self.vector_dim is None:
                self.vector_dim = vector.shape[0]
                self._initialize_vectors_file()
                self._save_dimension()
                logger.info(f"Initialized vector dimension to {self.vector_dim}")

            text_hash = self._hash_text(text)
            if text_hash in self.hash_to_index:
                logger.warning(
                    "Hash collision or duplicate text. Overwriting existing vector."
                )
                index = self.hash_to_index[text_hash]
            else:
                index = len(self.hash_to_index)
                if index >= len(self.vectors):
                    self._double_vectors_file()
                self.hash_to_index[text_hash] = index

            self.vectors[index] = vector
            logger.debug(
                f"Added new text-vector pair. Total pairs: {len(self.hash_to_index)}"
            )
        except Exception as e:
            logger.error(f"Error adding text-vector pair: {str(e)}")
            raise

    def _initialize_vectors_file(self):
        if self.vector_dim is None:
            logger.info("Vector dimension not set. Waiting for first add() call.")
            return

        if not self.vectors_file.exists():
            logger.info(
                f"Creating initial vectors file with {self.initial_vectors} vectors"
            )
            self.vectors = np.memmap(
                self.vectors_file,
                dtype="float32",
                mode="w+",
                shape=(self.initial_vectors, self.vector_dim),
            )
        else:
            self.vectors = np.memmap(self.vectors_file, dtype="float32", mode="r+")
            self.vectors = self.vectors.reshape(-1, self.vector_dim)
        logger.info(f"Vectors file initialized with shape: {self.vectors.shape}")

    def _double_vectors_file(self):
        current_size = len(self.vectors)
        new_size = current_size * 2
        logger.info(f"Doubling vectors file from {current_size} to {new_size} vectors")
        self.vectors.flush()
        new_vectors = np.memmap(
            self.vectors_file,
            dtype="float32",
            mode="r+",
            shape=(new_size, self.vector_dim),
        )
        new_vectors[:current_size] = self.vectors[:]
        self.vectors = new_vectors

    def _save_dimension(self):
        with open(self.dimension_file, "w") as f:
            f.write(str(self.vector_dim))
        logger.info(
            f"Saved vector dimension {self.vector_dim} to {self.dimension_file}"
        )

    def _load_dimension(self):
        if self.dimension_file.exists():
            with open(self.dimension_file) as f:
                self.vector_dim = int(f.read().strip())
            logger.info(
                f"Loaded vector dimension {self.vector_dim} from {self.dimension_file}"
            )
        else:
            logger.warning(
                "Dimension file not found. Vector dimension remains uninitialized."
            )

    def save(self) -> None:
        try:
            if self.vectors is not None:
                self.vectors.flush()

            # Convert hash_to_index dict to a format suitable for JSON
            # JSON doesn't support integer keys, so we keep everything as strings
            serializable_index = {
                str(hash_): int(index)  # Ensure indices are serialized as integers
                for hash_, index in self.hash_to_index.items()
            }

            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(serializable_index, f, indent=2)

            self._save_dimension()
            logger.info(f"Saved TextVectorMap to {self.directory}")
        except Exception as e:
            logger.error(f"Error saving TextVectorMap: {str(e)}")
            raise

    def load(self, name: str = None) -> None:
        name_details = name if name else ""
        try:
            self._load_dimension()
            if self.index_file.exists() and self.vectors_file.exists():
                with open(self.index_file, encoding="utf-8") as f:
                    # Load and convert the JSON data back to the expected format
                    loaded_index = json.load(f)
                    self.hash_to_index = {
                        str(hash_): int(index)  # Ensure we maintain the correct types
                        for hash_, index in loaded_index.items()
                    }

                if self.vector_dim is not None:
                    self.vectors = np.memmap(
                        self.vectors_file, dtype="float32", mode="r+"
                    )
                    self.vectors = self.vectors.reshape(-1, self.vector_dim)
                    logger.info(f"Loaded vectors file with shape: {self.vectors.shape}")
                else:
                    logger.warning(
                        "Vector dimension not set. Unable to load vectors file."
                    )

                logger.info(
                    f"Loaded TextVectorMap ({name_details}) from {self.directory}"
                )
            else:
                logger.warning(
                    f"No existing files found. Initialized empty TextVectorMap ({name_details})."
                )
        except Exception as e:
            logger.error(f"Error loading TextVectorMap ({name_details}): {str(e)}")
            raise

    def get_vector(self, text: str) -> Optional[np.ndarray]:
        try:
            text_hash = self._hash_text(text)
            if text_hash not in self.hash_to_index:
                logger.debug(f"Text hash not found in index: {text_hash}")
                return None
            index = self.hash_to_index[text_hash]
            return self.vectors[index]
        except Exception as e:
            logger.error(f"Error retrieving vector for text: {str(e)}")
            raise

    def __contains__(self, text: str) -> bool:
        return self._hash_text(text) in self.hash_to_index

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "vectors") and self.vectors is not None:
            self.vectors.flush()
            del self.vectors
            self.vectors = None
        logger.info(f"Closed TextVectorMap in directory: {self.directory}")


class CachedEmbeddingWrapper(Wrapper, Encoder):
    def __init__(self, model: Encoder, cache_path: Union[str | Path]):
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "encode"):
            self.cache = TextVectorMap(self.cache_path / "cache")
            self.cache.load(name="cache")
            self._wrap_encode_method()
        else:
            logger.error("Model must have an 'encode' method.")
            raise ValueError("Invalid model encoding method")

        logger.info("Initialized CachedEmbeddingWrapper")

    def _wrap_encode_method(self):
        original_encode = self._model.encode

        def wrapped_encode(
            texts: List[str], batch_size: int = 32, **kwargs
        ) -> np.ndarray:
            return self._cached_encode(
                texts, self.cache, original_encode, batch_size, **kwargs
            )

        self._model.encode = wrapped_encode

    def _cached_encode(
        self,
        texts: List[str],
        cache: TextVectorMap,
        encode_func: Callable[[List[str]], np.ndarray],
        batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        try:
            results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                vector = cache.get_vector(text)
                if vector is not None:
                    results.append(vector)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            if uncached_texts:
                logger.info(f"Encoding {len(uncached_texts)} new texts")
                new_vectors = encode_func(uncached_texts, **kwargs)
                if isinstance(new_vectors, torch.Tensor):
                    new_vectors = new_vectors.cpu().numpy()
                for text, vector in zip(uncached_texts, new_vectors):
                    cache.add(text, vector)
                results.extend(new_vectors)
                cache.save()
            else:
                logger.info("All texts found in cache")

            final_results = [None] * len(texts)
            uncached_idx = 0
            for i in range(len(texts)):
                if i in uncached_indices:
                    final_results[i] = results[
                        len(texts) - len(uncached_texts) + uncached_idx
                    ]
                    uncached_idx += 1
                else:
                    final_results[i] = results[i - uncached_idx]

            return np.array(final_results)
        except Exception as e:
            logger.error(f"Error in cached encoding: {str(e)}")
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __dir__(self) -> List[str]:
        return list(set(super().__dir__() + dir(self._model)))

    def __del__(self):
        self.close()

    def close(self):
        self.cache.close()
        logger.info("Closed CachedEmbeddingWrapper")
