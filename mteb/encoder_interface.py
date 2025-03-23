from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

from torch.utils.data import DataLoader

from mteb.types import Array, BatchedInput, PromptType

if TYPE_CHECKING:
    from mteb.abstasks import TaskMetadata

Corpus = Union[list[dict[str, str]], dict[str, list[str]]]


@runtime_checkable
class Encoder(Protocol):
    """The interface for an encoder in MTEB.

    Besides the required functions specified below, the encoder can additionally specify the the following signatures seen below.
    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, model: str, revision: str, **kwargs) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            model: name of the model
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
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        ...

    def combine_query_and_instruction(
        self,
        query: str,
        instruction: str,
    ) -> str:
        """Combines a query with an instruction.

        Args:
            query: The query text to combine.
            instruction: The instruction text to combine with the query.

        Returns:
            The combined query and instruction text.
        """
        ...

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute the similarity between two collections of embeddings. The output will be a matrix with the similarity scores between all embeddings
        from the first parameter and all embeddings from the second parameter. This differs from similarity_pairwise which computes the similarity
        between each pair of embeddings.

        read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity

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
        """Compute the similarity between two collections of embeddings. The output will be a vector with the similarity scores between each pair of
        embeddings.

        read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity_pairwise

        Args:
            embeddings1: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            A [num_embeddings]-shaped torch tensor with pairwise similarity scores.
        """
        ...

    # def predict(
    #     self,
    #     queries: Sequence[str],
    #     passages: Sequence[str],
    #     *,
    #     task_name: str | None = None,
    #     instruction: str | None = None,
    #     **kwargs: Any,
    # ) -> Array:
    #     """Predicts relevance scores for query-passage pairs. Note that, unlike the encoder, the cross-encoder can compare across queries and passages.
    #
    #     Args:
    #         queries: The queries to score.
    #         passages: The passages to score.
    #         task_name: The name of the task to score.
    #         instruction: Optional instruction text to combine with the query.
    #         **kwargs: Additional arguments to pass to the cross-encoder.
    #
    #     Returns:
    #         The predicted relevance scores for each query-passage pair.
    #     """
    #     ...
