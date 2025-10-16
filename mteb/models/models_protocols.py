from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import (
    Array,
    BatchedInput,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

if TYPE_CHECKING:
    from mteb.models.model_meta import ModelMeta


@runtime_checkable
class SearchProtocol(Protocol):
    """Interface for searching models."""

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
        """
        ...

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus using the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        ...

    @property
    def mteb_model_meta(self) -> "ModelMeta":
        """Metadata of the model"""
        ...


@runtime_checkable
class EncoderProtocol(Protocol):
    """The interface for an encoder in MTEB.

    Besides the required functions specified below, the encoder can additionally specify the following signatures seen below.
    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, model_name: str, revision: str | None, **kwargs: Any) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            model_name: Name of the model
            revision: revision of the model
            kwargs: Any additional kwargs
        """
        ...

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
        """Encodes the given sentences using the encoder.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task. Encoders (e.g. SentenceTransformers) use to
                select the appropriate prompts, with priority given to more specific task/prompt combinations over general ones.

                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        ...

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute the similarity between two collections of embeddings.

        The output will be a matrix with the similarity scores between all embeddings from the first parameter and all
        embeddings from the second parameter. This differs from similarity_pairwise which computes the similarity
        between corresponding pairs of embeddings.

        Read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity

        Args:
            embeddings1: [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2: [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.
        """
        ...

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute the similarity between two collections of embeddings. The output will be a vector with the similarity scores between each pair of embeddings.

        Read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity_pairwise

        Args:
            embeddings1: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            A [num_embeddings]-shaped torch tensor with pairwise similarity scores.
        """
        ...

    @property
    def mteb_model_meta(self) -> "ModelMeta":
        """Metadata of the model"""
        ...


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """The interface for a CrossEncoder in MTEB.

    Besides the required functions specified below, the cross-encoder can additionally specify the following signatures seen below.
    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, model_name: str, revision: str | None, **kwargs: Any) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            model_name: Name of the model
            revision: revision of the model
            kwargs: Any additional kwargs
        """
        ...

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
        """Predicts relevance scores for pairs of inputs. Note that, unlike the encoder, the cross-encoder can compare across inputs.

        Args:
            inputs1: First Dataloader of inputs to encode. For reranking tasks, these are queries (for text only tasks `QueryDatasetType`).
            inputs2: Second Dataloader of inputs to encode. For reranking, these are documents (for text only tasks `RetrievalOutputType`).
            task_metadata: Metadata of the current task.
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the cross-encoder.

        Returns:
            The predicted relevance scores for each inputs pair.
        """
        ...

    @property
    def mteb_model_meta(self) -> "ModelMeta":
        """Metadata of the model"""
        ...


MTEBModels = EncoderProtocol | CrossEncoderProtocol | SearchProtocol
"""Type alias for all MTEB model types as many models implement multiple protocols and many tasks can be solved by multiple model types."""
