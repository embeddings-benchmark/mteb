from __future__ import annotations

from typing import TYPE_CHECKING

from .model_meta import ModelMeta
from .models_protocols import (
    CrossEncoderProtocol,
    EncoderProtocol,
    MTEBModels,
    SearchProtocol,
)

if TYPE_CHECKING:
    from .cache_wrappers import CacheBackendProtocol, CachedEmbeddingWrapper
    from .compression_wrappers import CompressionWrapper
    from .hybrid_wrappers import HybridSearch
    from .openai_wrappers import (
        OpenAIAPIEncodeWrapper,
        OpenAIAPIRerankWrapper,
        OpenAIAPITokenEmbedWrapper,
    )
    from .search_encoder_index.search_backend_protocol import (
        IndexEncoderSearchProtocol,
    )
    from .search_wrappers import SearchCrossEncoderWrapper, SearchEncoderWrapper
    from .sentence_transformer_wrapper import (
        CrossEncoderWrapper,
        SentenceTransformerEncoderWrapper,
        SparseEncoderWrapper,
        sentence_transformers_loader,
    )

# Every wrapper below imports torch. Importing a submodule runs this `__init__` first, so eager
# re-exports here put torch on the import path of anything touching `mteb.models` including
# `mteb.abstasks`, and through it results analysis, the leaderboard and the API. Resolving them on
# first access keeps that path torch-free while `from mteb.models import X` keeps working.
_LAZY_ATTRIBUTES: dict[str, str] = {
    "CacheBackendProtocol": ".cache_wrappers",
    "CachedEmbeddingWrapper": ".cache_wrappers",
    "CompressionWrapper": ".compression_wrappers",
    "CrossEncoderWrapper": ".sentence_transformer_wrapper",
    "HybridSearch": ".hybrid_wrappers",
    "IndexEncoderSearchProtocol": ".search_encoder_index.search_backend_protocol",
    "OpenAIAPIEncodeWrapper": ".openai_wrappers",
    "OpenAIAPIRerankWrapper": ".openai_wrappers",
    "OpenAIAPITokenEmbedWrapper": ".openai_wrappers",
    "SearchCrossEncoderWrapper": ".search_wrappers",
    "SearchEncoderWrapper": ".search_wrappers",
    "SentenceTransformerEncoderWrapper": ".sentence_transformer_wrapper",
    "SparseEncoderWrapper": ".sentence_transformer_wrapper",
    "sentence_transformers_loader": ".sentence_transformer_wrapper",
}


def __getattr__(name: str) -> object:
    """Import the wrapper modules on first access rather than at package import time."""
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache it, so later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CacheBackendProtocol",
    "CachedEmbeddingWrapper",
    "CompressionWrapper",
    "CrossEncoderProtocol",
    "CrossEncoderWrapper",
    "EncoderProtocol",
    "HybridSearch",
    "IndexEncoderSearchProtocol",
    "MTEBModels",
    "ModelMeta",
    "OpenAIAPIEncodeWrapper",
    "OpenAIAPIRerankWrapper",
    "OpenAIAPITokenEmbedWrapper",
    "SearchCrossEncoderWrapper",
    "SearchEncoderWrapper",
    "SearchProtocol",
    "SentenceTransformerEncoderWrapper",
    "SparseEncoderWrapper",
    "sentence_transformers_loader",
]
