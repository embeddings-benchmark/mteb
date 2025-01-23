from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

Corpus = Union[list[dict[str, str]], dict[str, list[str]]]


class PromptType(str, Enum):
    query = "query"
    passage = "passage"


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
        self.device = device

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


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
class EncoderWithConversationEncode(Encoder, Protocol):
    """The optional interface for an encoder that supports encoding conversations."""

    def encode_conversations(
        self,
        conversations: Sequence[Sequence[str]],
        *,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given conversations using the encoder.

        Args:
            conversations: The conversations to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)

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


class ImageEncoder:
    """Interface for image encoder designed based on VLM2VecWrapper.
    There is not a perfect 1-1 match, e.g. device can be None here.
    The intention here is to define the current interface and adapt to as close to MTEB as possible
    and align as much as possible with sentencetransformers.
    """

    def __init__(
        self,
        device: str | None,
        **kwargs: Any,
    ):
        pass

    def encode(  # current a 1-1 match with Encoder.encode
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        pass

    def get_image_embeddings(  # Seems like sentence transformers use a singular encode for both images and text. Not sure if we want to do the same.
        # If not it might be ideal to redefine Encoder.encode
        self,
        images: list[Image.Image] | DataLoader,
        **kwargs,
        # removed batch_size, it is not required that it will accept kwargs
    ) -> np.ndarray:  # added standard output (I believe we actually expect tensors in the code, but would like to be consistent)
        pass

    def get_text_embeddings(  # any reason for this?
        self,
        texts: list[str],
        **kwargs,
    ) -> np.ndarray:
        pass

    def get_fused_embeddings(  # hmm what if I have a document with images at specific positions?
        self,
        texts: list[str] | None = None,
        images: list[Image.Image]
        | DataLoader
        | None = None,  # the requirement for these two to be the same seems odd (docs without images, images without associated text, docs with multiple images)
        # fusion_mode: str="sum", # will remove this as it should be required in the interface
        **kwargs: Any,
    ) -> np.ndarray:
        pass
