from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from datasets import Dataset
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from PIL import Image


# --- Output types ---
Array = np.ndarray | torch.Tensor
"""General array type, can be a numpy array or a torch tensor."""


# --- Input types ---
class PromptType(str, Enum):
    """The type of prompt used in the input for retrieval models. Used to differentiate between queries and documents.

    Attributes:
        query: A prompt that is a query.
        document: A prompt that is a document.
    """

    query = "query"
    document = "document"


class ConversationTurn(TypedDict):
    """A conversation, consisting of a list of messages.

    Attributes:
        role: The role of the message sender.
        content: The content of the message.
    """

    role: str
    content: str


Conversation = list[ConversationTurn]
"""A conversation, consisting of a list of messages."""


class TextInput(TypedDict):
    """The input to the encoder for text.

    Attributes:
        text: The text to encode. Can be a list of texts or a list of lists of texts.
    """

    text: list[str]


class CorpusInput(TextInput):
    """The input to the encoder for retrieval corpus.

    Attributes:
        title: The title of the text to encode. Can be a list of titles or a
            list of lists of titles.
        body: The body of the text to encode. Can be a list of bodies or a
            list of lists of bodies.
    """

    title: list[str]
    body: list[str]


class QueryInput(TextInput):
    """The input to the encoder for queries.

    Attributes:
        query: The query to encode. Can be a list of queries or a list of lists of queries.
        conversation: Optional. A list of conversations, each conversation is a list of messages.
        instruction: Optional. A list of instructions to encode.
    """

    query: list[str]
    conversation: NotRequired[list[Conversation]]
    instruction: NotRequired[list[str]]


class ImageInput(TypedDict):
    """The input to the encoder for images.

    Attributes:
        image: The image to encode. Can be a list of images or a list of lists of images.
    """

    image: list[Image.Image]


class AudioInput(TypedDict):
    """The input to the encoder for audio.

    Attributes:
        audio: The audio to encode. Can be a list of audio files or a list of lists of audio files.
    """

    audio: list[list[bytes]]


class MultimodalInput(TextInput, CorpusInput, QueryInput, ImageInput, AudioInput):  # type: ignore[misc]
    """The input to the encoder for multimodal data."""

    pass


BatchedInput = (
    TextInput | CorpusInput | QueryInput | ImageInput | AudioInput | MultimodalInput
)
"""
Represents the input format accepted by the encoder for a batch of data.

The encoder can process several input types depending on the task or modality.
Each type is defined as a separate structured input with its own fields.

### Supported input types

1. **[`TextInput`][mteb.types._encoder_io.TextInput]**
   For pure text inputs.

   ```python
   {"text": ["This is a sample text.", "Another text."]}
   ```
2. **[`CorpusInput`][mteb.types._encoder_io.CorpusInput]**
   For corpus-style inputs with titles and bodies.

   ```python
   {"text": ["Title 1 Body 1", "Title 2 Body 2"], "title": ["Title 1", "Title 2"], "body": ["Body 1", "Body 2"]}
   ```
3. **[`QueryInput`][mteb.types._encoder_io.QueryInput]**
   For query–instruction pairs, typically used in retrieval or question answering tasks. Queries and instructions are combined with the model's instruction template.

   ```python
   {
       "text": ["Instruction: Your task is to find document for this query. Query: What is AI?", "Instruction: Your task is to find term for definition. Query: Define machine learning."],
       "query": ["What is AI?", "Define machine learning."],
       "instruction": ["Your task is find document for this query.", "Your task is to find term for definition."]
   }
   ```
4. **[`ImageInput`][mteb.types._encoder_io.ImageInput]**
   For visual inputs consisting of images.

   ```python
   {"image": [PIL.Image1, PIL.Image2]}
   ```
5. **[`MultimodalInput`][mteb.types._encoder_io.MultimodalInput]**
   For combined text–image (multimodal) inputs.

   ```python
   {"text": ["This is a sample text."], "image": [PIL.Image1]}
   ```
"""


TextBatchedInput = TextInput | CorpusInput | QueryInput
"""The input to the encoder for a batch of text data."""

QueryDatasetType = Dataset
"""Retrieval query dataset, containing queries. Should have columns `id`, `text`."""
CorpusDatasetType = Dataset
"""Retrieval corpus dataset, containing documents. Should have columns `id`, `title`, `body`."""
InstructionDatasetType = Dataset
"""Retrieval instruction dataset, containing instructions. Should have columns `query-id`, `instruction`."""
RelevantDocumentsType = Mapping[str, Mapping[str, int]]
"""Relevant documents for each query, mapping query IDs to a mapping of document IDs and their relevance
scores. Should have columns `query-id`, `corpus-id`, `score`."""
TopRankedDocumentsType = Mapping[str, list[str]]
"""Top-ranked documents for each query, mapping query IDs to a list of document IDs. Should
have columns `query-id`, `corpus-ids`."""

RetrievalOutputType = dict[str, dict[str, float]]
"""Retrieval output, containing the scores for each query-document pair."""
