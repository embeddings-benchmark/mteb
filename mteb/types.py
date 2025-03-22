from __future__ import annotations

from enum import Enum
from typing import TypedDict

import numpy as np
import torch
from PIL import Image

Array = np.ndarray | torch.Tensor


class PromptType(str, Enum):
    query = "query"
    passage = "passage"


class Conversation(TypedDict):
    """A conversation, consisting of a list of messages.

    Args:
        role: The role of the message sender.
        content: The content of the message.
    """

    role: str
    content: str


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
    # list[list[str]] and list[Conversation] is used for conversations datasets
    query: list[str] | list[list[str]] | list[Conversation]
    instruction: list[str]
