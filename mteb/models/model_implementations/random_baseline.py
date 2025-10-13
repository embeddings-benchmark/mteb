from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types._encoder_io import Array, BatchedInput, PromptType


class RandomBaseline:
    """A random baseline that generates random embeddings. Useful to establish a lower bound for embedding performance.
    The embeddings are conditioned on the input text, so that the same text always gets the same embedding.

    This implements the Encoder interface.
    """

    mteb_model_meta: ModelMeta | None = None

    def __init__(self, model_name: str, revision: str | None, **kwargs: Any) -> None:
        self.rng_state = np.random.RandomState(42)
        self.embedding_dim = 32  # not sure it matters what dimension we use here

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
        all_embeddings = []
        for batch in inputs:
            if "text" in batch and "image" in batch:
                for text, image in zip(batch["text"], batch["image"]):
                    text_vector = self._string_to_vector(text, self.embedding_dim)
                    image_vector = self._image_to_vector(image, self.embedding_dim)
                    combined_vector = (text_vector + image_vector) / 2
                    all_embeddings.append(combined_vector)
            elif "image" in batch:
                for image in batch["image"]:
                    vector = self._image_to_vector(image, self.embedding_dim)
                    all_embeddings.append(vector)
            elif "text" in batch:
                for text in batch["text"]:
                    vector = self._string_to_vector(text, self.embedding_dim)
                    all_embeddings.append(vector)
            else:
                raise KeyError("Input batch must contain 'text' and/or 'image' keys.")
        return np.vstack(all_embeddings)

    @staticmethod
    def _string_to_vector(s: str | None, size: int):
        """Generate a deterministic random vector based on a string."""
        if s is None:
            s = ""
        # Convert string to a numeric seed
        seed = int.from_bytes(s.encode("utf-8"), byteorder="big") % (
            2**32
        )  # numpy rng seed must be between 0 and 2**32
        rng = np.random.default_rng(seed)
        return rng.random(size)

    @staticmethod
    def _image_to_vector(image: Image.Image, size: int):
        """Generate a deterministic random vector based on image content."""
        # Convert image to bytes and then to a numeric seed
        image_bytes = image.tobytes()
        seed = int.from_bytes(image_bytes, byteorder="big") % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(size)

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Cosine similarity"""
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        normalized1 = embeddings1 / (norm1 + 1e-10)
        normalized2 = embeddings2 / (norm2 + 1e-10)
        return np.dot(normalized1, normalized2.T)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Cosine similarity"""
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        normalized1 = embeddings1 / (norm1 + 1e-10)
        normalized2 = embeddings2 / (norm2 + 1e-10)
        return np.sum(normalized1 * normalized2, axis=1)


random_baseline = ModelMeta(
    loader=RandomBaseline,  # type: ignore
    name="mteb/random-baseline",
    modalities=["text", "image"],
    languages=None,
    open_weights=True,
    revision="1",
    release_date=None,
    n_parameters=0,
    memory_usage_mb=0,
    embed_dim=32,
    license="mit",
    max_tokens=np.inf,
    reference=None,
    similarity_fn_name="cosine",  # type: ignore
    framework=[],
    use_instructions=False,
    public_training_code=None,  # No training code, as this is a random baseline
    public_training_data=None,  # No training data, as this is a random baseline
    training_datasets=set(),
)
