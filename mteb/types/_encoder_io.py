from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypedDict, Union

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

# --- Output types ---
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


class BatchedInput(TypedDict, total=False):
    """The input to the encoder. This is the input to the encoder when using the encode function.


    Args:
        text: The text to encode.
        image: The image to encode. Can be a list of images or a list of lists of images.
        audio: The audio to encode. Can be a list of audio files or a list of lists of audio files.

        Retrieval corpus:
            title: The title of the text to encode.
            body: The body of the text to encode.

        Retrieval query:
            query: The query to encode.
            instruction: The instruction to encode.
    """

    text: list[str]
    image: list[list[Image.Image]]
    audio: list[list[bytes]]
    # Retrieval corpus
    title: list[str]
    body: list[str]
    # Retrieval query
    query: list[str]
    conversation: list[Conversation]
    instruction: list[str]


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
