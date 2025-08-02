from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypedDict, Union

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from typing_extensions import NotRequired

# --- Output types ---
# should be as Union, because `|` won't work for python3.9
Array = Union[np.ndarray, torch.Tensor]


# --- Input types ---
class PromptType(str, Enum):
    query = "query"
    passage = "passage"


class ConversationTurn(TypedDict):
    """A conversation, consisting of a list of messages.

    Args:
        role: The role of the message sender.
        content: The content of the message.
    """

    role: str
    content: str


Conversation = list[ConversationTurn]


class TextInput(TypedDict):
    """The input to the encoder for text.

    Args:
        text: The text to encode. Can be a list of texts or a list of lists of texts.
    """

    text: list[str]


class CorpusInput(TextInput):
    """The input to the encoder for retrieval corpus.

    Args:
        title: The title of the text to encode. Can be a list of titles or a
            list of lists of titles.
        body: The body of the text to encode. Can be a list of bodies or a
            list of lists of bodies.
    """

    title: list[str]
    body: list[str]


class QueryInput(TextInput):
    """The input to the encoder for queries.

    Args:
        query: The query to encode. Can be a list of queries or a list of lists of queries.
        conversation: Optional. A list of conversations, each conversation is a list of messages.
        instruction: Optional. A list of instructions to encode.
    """

    query: list[str]
    conversation: NotRequired[list[Conversation]]
    instruction: NotRequired[list[str]]


class ImageInput(TypedDict):
    """The input to the encoder for images.

    Args:
        image: The image to encode. Can be a list of images or a list of lists of images.
    """

    image: list[list[Image.Image]]


class AudioInput(TypedDict):
    """The input to the encoder for audio.

    Args:
        audio: The audio to encode. Can be a list of audio files or a list of lists of audio files.
    """

    audio: list[list[bytes]]


class MultimodalInput(TextInput, CorpusInput, QueryInput, ImageInput, AudioInput):
    """The input to the encoder for multimodal data."""

    pass


# should be as Union, because `|` won't work for python3.9
BatchedInput = Union[
    TextInput, CorpusInput, QueryInput, ImageInput, AudioInput, MultimodalInput
]

TextBatchedInput = Union[TextInput, CorpusInput, QueryInput]


QueryDatasetType = Dataset
"""Retrieval query dataset, containing queries. Should have columns `id`, `text`."""
CorpusDatasetType = Dataset
"""Retrieval corpus dataset, containing documents. Should have columns `id`, `title`, `body`."""
InstructionDatasetType = Dataset
"""Retrieval instruction dataset, containing instructions. Should have columns `query-id`, `instruction`."""
RelevantDocumentsType = Mapping[str, Mapping[str, float]]
"""Relevant documents for each query, mapping query IDs to a mapping of document IDs and their relevance
scores. Should have columns `query-id`, `corpus-id`, `score`."""
TopRankedDocumentsType = Mapping[str, list[str]]
"""Top-ranked documents for each query, mapping query IDs to a list of document IDs. Should
have columns `query-id`, `corpus-ids`."""
