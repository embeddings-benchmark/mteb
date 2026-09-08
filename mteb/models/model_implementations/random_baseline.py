from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_audio_dependencies,
    requires_image_dependencies,
)
from mteb.models.modality_collators import VideoCollator
from mteb.models.model_meta import ModelMeta
from mteb.similarity_functions import (
    max_sim,
    select_pairwise_similarity,
    select_similarity,
)
from mteb.types import PromptType

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types._encoder_io import (
        Array,
        AudioInputItem,
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


def _text_to_bytes(text: str | None) -> bytes:
    """Convert a (possibly missing) text sample into bytes for deterministic seeding."""
    return (text or "").encode("utf-8")


def _image_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image sample into bytes for deterministic seeding."""
    return image.tobytes()


def _audio_to_bytes(audio: AudioInputItem) -> bytes:
    """Convert an audio sample into bytes for deterministic seeding."""
    return audio["array"].tobytes()


def _video_to_bytes(item: torch.Tensor) -> bytes:
    """Convert a video frames tensor into bytes for deterministic seeding."""
    return item.numpy().tobytes()


def _bytes_to_vector(data: bytes, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on raw bytes.

    Args:
        data: Input bytes.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    # numpy rng seed must be between 0 and 2**32
    seed = int(hashlib.sha256(data).hexdigest(), 16) % 2**32
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def _bytes_to_multi_vector(
    data: bytes, num_tokens: int, size: int
) -> NDArray[np.floating]:
    """Generate a deterministic sequence of unit-norm token vectors based on raw bytes.

    Mimics a late-interaction (ColBERT-style) encoder, which represents each input
    as one embedding per token rather than a single pooled vector.

    Args:
        data: Input bytes.
        num_tokens: Number of token vectors to generate.
        size: Dimensionality of each token vector.

    Returns:
        A numpy array of shape (num_tokens, size).
    """
    vectors = np.stack(
        [
            _bytes_to_vector(data + i.to_bytes(4, "little"), size)
            for i in range(num_tokens)
        ]
    )
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / (norms + 1e-10)).astype(np.float32)


def _string_to_vector(text: str | None, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on a string.

    Args:
        text: Input string.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    return _bytes_to_vector(_text_to_bytes(text), size)


def _image_to_vector(image: Image.Image, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on image content.

    Args:
        image: PIL Image object.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    return _bytes_to_vector(_image_to_bytes(image), size)


def _audio_to_vector(audio: AudioInputItem, size: int) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on audio content.

    Args:
        audio: Audio data (e.g., numpy array).
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    return _bytes_to_vector(_audio_to_bytes(audio), size)


def _video_to_vector(
    item: torch.Tensor,
    size: int,
) -> NDArray[np.floating]:
    """Generate a deterministic random vector based on video content.

    Args:
        item: Video frames tensor.
        size: Size of the output vector.

    Returns:
        A numpy array of shape (size,) containing the random vector.
    """
    return _bytes_to_vector(_video_to_bytes(item), size)


def _attach_modality_collator(
    inputs: DataLoader[BatchedInput],
    *,
    fps: float | None,
    max_frames: int | None,
    num_frames: int | None,
) -> None:
    """Attach a VideoCollator to `inputs` if its dataset exposes video or audio features.

    Args:
        inputs: DataLoader whose dataset may contain 'video' and/or 'audio' features.
        fps: Target frames per second for video sampling.
        max_frames: Safety cap on frames per video for FPS mode.
        num_frames: If set, use fixed-sample mode instead of FPS-based.
    """
    has_video = "video" in inputs.dataset.features
    has_audio = "audio" in inputs.dataset.features
    if has_video or has_audio:
        requires_audio_dependencies()
        requires_image_dependencies()
        inputs.collate_fn = VideoCollator(
            target_sampling_rate=16000,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
        )


_EMBEDDING_DIM = 32

_common_mock_metadata = dict(
    languages=None,
    open_weights=True,
    revision="1",
    release_date=None,
    n_parameters=0,
    n_embedding_parameters=0,
    memory_usage_mb=0,
    license="mit",
    max_tokens=np.inf,
    reference=None,
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


def _sparsify_vector(
    vector: NDArray[np.floating], n_nonzero: int
) -> NDArray[np.floating]:
    """Zero out all but the `n_nonzero` largest-magnitude entries of a vector, keeping its shape.

    Mimics the output of a lexical/sparse encoder (e.g. SPLADE), which only assigns non-zero
    importance weights to a small subset of the vocabulary. Unlike Matryoshka-style dense embedding
    truncation (which shrinks a vector to its first `k` dimensions), this keeps the full
    dimensionality and zeroes out everything except the `n_nonzero` largest values, wherever they
    are.

    Args:
        vector: Dense input vector.
        n_nonzero: Number of entries to keep non-zero.

    Returns:
        A copy of `vector`, same shape, with all but the `n_nonzero` largest entries set to 0.
    """
    if n_nonzero >= vector.size:
        return vector
    vector = vector.copy()
    zero_idx = np.argpartition(vector, -n_nonzero)[:-n_nonzero]
    vector[zero_idx] = 0.0
    return vector


_SPARSE_EMBEDDING_DIM = 128
_SPARSE_N_NONZERO = 16


def _batch_to_sparse_embeddings(
    inputs: DataLoader[BatchedInput], embedding_dim: int, n_nonzero: int
) -> NDArray[np.floating]:
    """Convert batched text/image/audio/video inputs into sparse (mostly-zero) embeddings.

    Args:
        inputs: A DataLoader yielding batches of inputs, where each batch is a dictionary
                that may contain 'text', 'image', 'audio', and/or 'video' keys.
        embedding_dim: The dimensionality of the output embeddings.
        n_nonzero: Number of non-zero entries to keep per embedding.

    Returns:
        A 2D numpy array of shape (num_samples, embedding_dim).
    """
    embeddings = _batch_to_embeddings(inputs, embedding_dim)
    return np.stack([_sparsify_vector(row, n_nonzero) for row in embeddings])


_LATE_INTERACTION_EMBEDDING_DIM = 16
_LATE_INTERACTION_NUM_TOKENS = 8


def _batch_to_multi_vector_embeddings(
    inputs: DataLoader[BatchedInput], num_tokens: int, embedding_dim: int
) -> NDArray[np.floating]:
    """Convert batched text/image/audio/video inputs into per-token (late-interaction) embeddings.

    Args:
        inputs: A DataLoader yielding batches of inputs, where each batch is a dictionary
                that may contain 'text', 'image', 'audio', and/or 'video' keys.
        num_tokens: Number of token vectors to generate per input.
        embedding_dim: Dimensionality of each token vector.

    Returns:
        A 3D numpy array of shape (num_samples, num_tokens, embedding_dim).
    """
    embeddings = []
    for batch in tqdm(inputs, desc="Encoding batches", unit="batch"):
        text_embeddings = []
        image_embeddings = []
        audio_embeddings = []
        video_embeddings = []

        if "text" in batch:
            text_embeddings = [
                _bytes_to_multi_vector(_text_to_bytes(txt), num_tokens, embedding_dim)
                for txt in batch["text"]
            ]
        if "image" in batch:
            image_embeddings = [
                _bytes_to_multi_vector(_image_to_bytes(img), num_tokens, embedding_dim)
                for img in batch["image"]
            ]
        if "audio" in batch:
            audio_embeddings = [
                _bytes_to_multi_vector(
                    _audio_to_bytes(audio), num_tokens, embedding_dim
                )
                for audio in batch["audio"]
            ]
        if "video" in batch:
            video_embeddings = [
                _bytes_to_multi_vector(
                    _video_to_bytes(video), num_tokens, embedding_dim
                )
                for video in batch["video"]
            ]

        max_len = max(
            [
                len(text_embeddings),
                len(image_embeddings),
                len(audio_embeddings),
                len(video_embeddings),
            ]
        )
        for i in range(max_len):
            combined_embedding = np.zeros((num_tokens, embedding_dim), dtype=np.float32)
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

    return np.stack(embeddings)


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
        embed_dim: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 10,
        **kwargs: Any,
    ) -> None:
        self.rng_state = np.random.default_rng(42)
        self.embedding_dim = embed_dim or _EMBEDDING_DIM
        self.array_framework = array_framework
        self.dtype = dtype
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

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
        _attach_modality_collator(
            inputs, fps=self.fps, max_frames=self.max_frames, num_frames=self.num_frames
        )
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
    embed_dim=[_EMBEDDING_DIM, 10],
    similarity_fn_name="cosine",
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
        has_video = "video" in inputs1.dataset.features
        has_audio = "audio" in inputs1.dataset.features
        if has_video or has_audio:
            collator = VideoCollator(
                target_sampling_rate=16000,
                fps=2.0,
            )
            inputs1.collate_fn = collator
            inputs2.collate_fn = collator

        embeddings1 = _batch_to_embeddings(inputs1, self.embedding_dim)
        embeddings2 = _batch_to_embeddings(inputs2, self.embedding_dim)
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2, strict=True):
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
    embed_dim=None,
    similarity_fn_name="cosine",
    **_common_mock_metadata,
)


class RandomSparseEncoderBaseline:
    """A random baseline that generates random sparse embeddings. Useful to establish a lower bound for sparse/lexical retrieval performance.
    The embeddings are conditioned on the input, so that the same input always gets the same embedding.
    Supports text, image, audio, and video inputs.

    This implements the Encoder interface.
    """

    mteb_model_meta: ModelMeta | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        embed_dim: int | None = None,
        n_nonzero: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 10,
        **kwargs: Any,
    ) -> None:
        self.rng_state = np.random.default_rng(42)
        self.embedding_dim = embed_dim or _SPARSE_EMBEDDING_DIM
        self.n_nonzero = n_nonzero or _SPARSE_N_NONZERO
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

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
        _attach_modality_collator(
            inputs, fps=self.fps, max_frames=self.max_frames, num_frames=self.num_frames
        )
        return _batch_to_sparse_embeddings(inputs, self.embedding_dim, self.n_nonzero)

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Dot product similarity between two sets of sparse embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Dot product similarity matrix between the two sets of embeddings
        """
        return select_similarity(
            embeddings1, embeddings2, self.mteb_model_meta.similarity_fn_name
        )

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Dot product similarity for pairs of sparse embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Dot product similarity for each pair of embeddings
        """
        return select_pairwise_similarity(
            embeddings1, embeddings2, self.mteb_model_meta.similarity_fn_name
        )


random_sparse_encoder_baseline = ModelMeta(
    loader=RandomSparseEncoderBaseline,
    name="mteb/baseline-random-sparse-encoder",
    model_type=["sparse"],
    embed_dim=_SPARSE_EMBEDDING_DIM,
    similarity_fn_name="dot",
    **_common_mock_metadata,
)


class RandomColBERTBaseline:
    """A random baseline that generates random per-token (late-interaction / ColBERT-style) embeddings.
    Useful to establish a lower bound for retrieval/reranking performance with late-interaction
    (multi-vector) models. The embeddings are conditioned on the input, so that the same input
    always gets the same embedding. Supports text, image, audio, and video inputs.

    This implements the SearchProtocol interface rather than the plain Encoder interface: real
    late-interaction models produce a variable number of per-token embeddings per document, which
    only supports retrieval-style search/reranking (via MaxSim over an indexed corpus), not the
    fixed-size-vector operations (indexing into a corpus-wide array, sklearn fit/predict, ...) that
    Classification/Clustering/etc. rely on.
    """

    mteb_model_meta: ModelMeta | None = None
    task_corpus: CorpusDatasetType | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        embed_dim: int | None = None,
        num_tokens: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 10,
        **kwargs: Any,
    ) -> None:
        self.rng_state = np.random.default_rng(42)
        self.embedding_dim = embed_dim or _LATE_INTERACTION_EMBEDDING_DIM
        self.num_tokens = num_tokens or _LATE_INTERACTION_NUM_TOKENS
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.task_corpus = None

    def _encode(self, inputs: DataLoader[BatchedInput]) -> Array:
        _attach_modality_collator(
            inputs, fps=self.fps, max_frames=self.max_frames, num_frames=self.num_frames
        )
        return _batch_to_multi_vector_embeddings(
            inputs, self.num_tokens, self.embedding_dim
        )

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use for dataloading.
        """
        self.task_corpus = corpus

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
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        """Search the indexed corpus using MaxSim similarity between multi-vector embeddings.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use for dataloading.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        from mteb._create_dataloaders import create_dataloader

        query_embeddings = self._encode(
            create_dataloader(
                queries,
                task_metadata=task_metadata,
                prompt_type=PromptType.query,
                num_proc=num_proc,
                **encode_kwargs,
            )
        )
        corpus_embeddings = self._encode(
            create_dataloader(
                self.task_corpus,
                task_metadata=task_metadata,
                prompt_type=PromptType.document,
                num_proc=num_proc,
                **encode_kwargs,
            )
        )
        query_ids = list(queries["id"])
        corpus_ids = list(self.task_corpus["id"])
        corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

        scores = max_sim(
            query_embeddings, corpus_embeddings
        )  # (num_queries, num_corpus)

        results: RetrievalOutputType = {qid: {} for qid in query_ids}
        for i, qid in enumerate(query_ids):
            candidate_ids = top_ranked[qid] if top_ranked is not None else corpus_ids
            candidate_idxs = torch.tensor(
                [corpus_id_to_idx[cid] for cid in candidate_ids]
            )
            candidate_scores = scores[i, candidate_idxs]
            top_n = min(top_k, len(candidate_ids))
            top_vals, top_idx = torch.topk(candidate_scores, top_n)
            for val, idx in zip(top_vals.tolist(), top_idx.tolist(), strict=True):
                results[qid][candidate_ids[idx]] = val

        self.task_corpus = None
        return results


random_colbert_baseline = ModelMeta(
    loader=RandomColBERTBaseline,
    name="mteb/baseline-random-colbert",
    model_type=["late-interaction"],
    embed_dim=_LATE_INTERACTION_EMBEDDING_DIM,
    similarity_fn_name="MaxSim",
    **_common_mock_metadata,
)
