from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._encoder_io import (
    BatchedInput,
    Conversation,
    ConversationTurn,
    CorpusDatasetType,
    EncodeKwargs,
    InstructionDatasetType,
    OutputDType,
    PromptType,
    QueryDatasetType,
    RelevantDocumentsType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)
from ._metadata import (
    ISOLanguage,
    ISOLanguageScript,
    ISOScript,
    Languages,
    Licenses,
    Modalities,
    ModelName,
    Revision,
)
from ._result import (
    HFSubset,
    RetrievalEvaluationResult,
    Score,
    ScoresDict,
    SplitName,
    SubmitResultsResponse,
)
from ._string_validators import StrDate, StrURL

if TYPE_CHECKING:
    from ._encoder_io import Array


def __getattr__(name: str) -> Any:
    """Forward `Array` to its lazy definition in `_encoder_io` (keeps `mteb.types` torch-free)."""
    if name == "Array":
        from ._encoder_io import Array

        return Array
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Array",
    "BatchedInput",
    "Conversation",
    "ConversationTurn",
    "CorpusDatasetType",
    "EncodeKwargs",
    "HFSubset",
    "ISOLanguage",
    "ISOLanguageScript",
    "ISOScript",
    "InstructionDatasetType",
    "Languages",
    "Licenses",
    "Modalities",
    "ModelName",
    "OutputDType",
    "PromptType",
    "QueryDatasetType",
    "RelevantDocumentsType",
    "RetrievalEvaluationResult",
    "RetrievalOutputType",
    "Revision",
    "Score",
    "ScoresDict",
    "SplitName",
    "StrDate",
    "StrURL",
    "SubmitResultsResponse",
    "TopRankedDocumentsType",
]
