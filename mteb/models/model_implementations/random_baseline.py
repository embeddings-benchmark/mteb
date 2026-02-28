from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import VideoCollator
from mteb.models.model_meta import ModelMeta
from mteb.similarity_functions import (
    select_pairwise_similarity,
    select_similarity,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchcodec.decoders import VideoDecoder

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types._encoder_io import (
        Array,
        AudioInputItem,
        BatchedInput,
        PromptType,
    )


def _string_to_vector(text: str | None, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on a string.

    Args:
        text: Input string.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    if text is None:
        text = ""
    # Convert string to a numeric seed
    # numpy rng seed must be between 0 and 2**32
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 2**32
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def _image_to_vector(image: Image.Image, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on image content.

    Args:
        image: PIL Image object.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    # Convert image to bytes and then to a numeric seed
    image_bytes = image.tobytes()
    seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def _audio_to_vector(audio: AudioInputItem, size: int) -> np.ndarray:
    """Generate a deterministic random vector based on audio content.

    Args:
        audio: Audio data (e.g., numpy array).
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    # Convert audio to bytes and then to a numeric seed
    audio_bytes = audio["array"].tobytes()
    seed = int(hashlib.sha256(audio_bytes).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def _video_to_vector(
    video: list[dict[str, VideoDecoder | AudioInputItem]],
    size: int,
) -> np.ndarray:
    """Generate a deterministic random vector based on video content.

    Args:
        video: Video data
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    # Convert video to bytes and then to a numeric seed
    video_bytes = b"".join(
        [
            VideoCollator.resample_video(item["frames"], 10).numpy().tobytes()
            + item["audio"]["array"].tobytes()
            for item in video
        ]
    )
    seed = int(hashlib.sha256(video_bytes).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


_EMBEDDING_DIM = 32

_common_mock_metadata = dict(
    languages=None,
    open_weights=True,
    revision="1",
    release_date=None,
    n_parameters=0,
    memory_usage_mb=0,
    embed_dim=_EMBEDDING_DIM,
    license="mit",
    max_tokens=np.inf,
    reference=None,
    similarity_fn_name="cosine",
    framework=[],
    use_instructions=False,
    public_training_code=None,  # No training code, as this is a random baseline
    public_training_data=None,  # No training data, as this is a random baseline
    training_datasets=set(),
    modalities=["text", "image", "audio", "video"],
)


def _batch_to_embeddings(
    inputs: DataLoader[BatchedInput], embedding_dim: int
) -> NDArray[np.floating]:
    """Convert batched text/image inputs into embeddings.

    Args:
        inputs: A DataLoader yielding batches of inputs, where each batch is a dictionary
                that may contain 'text' and/or 'image' keys.
        embedding_dim: The dimensionality of the output embeddings.

    Returns:
        A 2D numpy array of shape (num_samples, embedding_dim) containing the embeddings
    """
    embeddings = []
    for batch in tqdm(inputs, desc="Encoding batches", unit="batch"):
        text_embeddings = []
        image_embeddings = []
        audio_embeddings = []
        video_embeddings = []

        if "text" in batch:
            text_embeddings = [
                _string_to_vector(txt, embedding_dim) for txt in batch["text"]
            ]
        if "image" in batch:
            image_embeddings = [
                _image_to_vector(img, embedding_dim) for img in batch["image"]
            ]
        if "audio" in batch:
            audio_embeddings = [
                _audio_to_vector(audio, embedding_dim) for audio in batch["audio"]
            ]
        if "video" in batch:
            video_embeddings = [
                _video_to_vector(
                    video,
                    embedding_dim,
                )
                for video in batch["video"]
            ]

        # Combine embeddings
        max_len = max(
            [
                len(text_embeddings),
                len(image_embeddings),
                len(audio_embeddings),
                len(video_embeddings),
            ]
        )
        for i in range(max_len):
            combined_embedding = np.zeros(embedding_dim, dtype=np.float32)
            count = 0
            for embeddings_list in [
                text_embeddings,
                image_embeddings,
                audio_embeddings,
                video_embeddings,
            ]:
                if i < len(embeddings_list):
                    combined_embedding += embeddings_list[i]
                    count += 1
            if count > 0:
                combined_embedding /= count
            embeddings.append(combined_embedding)

    return np.vstack(embeddings)


class RandomEncoderBaseline:
    """A random baseline that generates random embeddings. Useful to establish a lower bound for embedding performance.
    The embeddings are conditioned on the input text, so that the same text always gets the same embedding.

    This implements the Encoder interface.
    """

    mteb_model_meta: ModelMeta | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        array_framework: Literal["numpy", "torch"] = "numpy",
        dtype: torch.dtype | np.floating = np.float32,
        embed_dim: int = _EMBEDDING_DIM,
        **kwargs: Any,
    ) -> None:
        self.rng_state = np.random.default_rng(42)
        self.embedding_dim = embed_dim
        self.array_framework = array_framework
        self.dtype = dtype

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
        embedding = _batch_to_embeddings(inputs, self.embedding_dim)
        if self.array_framework == "torch":
            return torch.tensor(embedding, dtype=self.dtype)
        return embedding.astype(self.dtype)

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Cosine similarity between two sets of embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Cosine similarity matrix between the two sets of embeddings
        """
        return select_similarity(
            embeddings1, embeddings2, self.mteb_model_meta.similarity_fn_name
        )

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Cosine similarity for pairs of embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Cosine similarity for each pair of embeddings
        """
        return select_pairwise_similarity(
            embeddings1, embeddings2, self.mteb_model_meta.similarity_fn_name
        )


random_encoder_baseline = ModelMeta(
    loader=RandomEncoderBaseline,
    name="mteb/baseline-random-encoder",
    model_type=["dense"],
    **_common_mock_metadata,
)


class RandomCrossEncoderBaseline:
    """A random baseline that generates random embeddings. Useful to establish a lower bound for embedding performance.
    The embeddings are conditioned on the input text, so that the same text always gets the same embedding.

    This implements the Encoder interface.
    """

    mteb_model_meta: ModelMeta | None = None

    def __init__(self, model_name: str, revision: str | None, **kwargs: Any) -> None:
        self.rng_state = np.random.default_rng(42)
        self.embedding_dim = _EMBEDDING_DIM

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        embeddings1 = _batch_to_embeddings(inputs1, self.embedding_dim)
        embeddings2 = _batch_to_embeddings(inputs2, self.embedding_dim)
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            normalized1 = emb1 / (norm1 + 1e-10)
            normalized2 = emb2 / (norm2 + 1e-10)
            similarities.append(np.dot(normalized1, normalized2))
        return np.array(similarities)


random_cross_encoder_baseline = ModelMeta(
    loader=RandomCrossEncoderBaseline,
    name="mteb/baseline-random-cross-encoder",
    model_type=["cross-encoder"],
    **_common_mock_metadata,
)
