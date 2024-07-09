from __future__ import annotations

from typing import Any, Dict, List, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import torch

Corpus = Union[List[Dict[str, str]], Dict[str, List[str]]]


@runtime_checkable
class Encoder(Protocol):
    """The interface for an encoder in MTEB.

    Besides the required functions specified below, the encoder can additionally specify the the following signatures seen below.
    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, device: str | None = None) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            device: The device to use for encoding. Can be ignored if the encoder is not using a device (e.g. for API)
        """

    def encode(
        self, sentences: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        ...


class EncoderWithSimilarity(Encoder, Protocol):
    """Besides the required functions in the Encoder interface, the encoder can additionally specify its own similiarity functions.

    MTEB will by default attempt to use similarity_pairwise function first before falling back to similarity function. If the encoder does not support
    similarity_pairwise function, it should simply not implement it.
    """

    def similarity(
        self,
        embeddings1: torch.Tensor | np.ndarray,
        embeddings2: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
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
        embeddings1: torch.Tensor | np.ndarray,
        embeddings2: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
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


@runtime_checkable
class EncoderWithQueryCorpusEncode(Encoder, Protocol):
    """The optional interface for an encoder that supports encoding queries and a corpus."""

    def encode_queries(
        self, queries: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given queries using the encoder.

        Args:
            queries: The queries to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded queries.
        """
        ...

    def encode_corpus(
        self, corpus: Corpus, *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given corpus using the encoder.

        Args:
            corpus: The corpus to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded corpus.
        """
        ...


@runtime_checkable
class EncoderWithConversationEncode(Encoder, Protocol):
    """The optional interface for an encoder that supports encoding conversations."""

    def encode_conversations(
        self,
        conversations: Sequence[Sequence[str]],
        *,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given conversations using the encoder.

        Args:
            conversations: The conversations to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded conversations.
        """
        ...

    @staticmethod
    def convert_conv_history_to_query(conversations: Sequence[Sequence[str]]) -> str:
        """Converts a conversation history to a single query.

        Args:
            conversations: The conversations to convert.

        Returns:
            The query.
        """
        ...
