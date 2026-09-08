from __future__ import annotations

from collections.abc import Callable

from mteb.models.model_meta import ModelMeta
from mteb.types import EncodeKwargs

EvaluationMetaResolver = Callable[[ModelMeta, EncodeKwargs], ModelMeta]
EvaluationMetaPredicate = Callable[[ModelMeta], bool]

_EVALUATION_META_RESOLVERS: list[
    tuple[EvaluationMetaPredicate, EvaluationMetaResolver]
] = []


def register_evaluation_meta_resolver(
    applies: EvaluationMetaPredicate, resolver: EvaluationMetaResolver
) -> None:
    """Register a model backend's evaluation metadata resolver."""
    _EVALUATION_META_RESOLVERS.append((applies, resolver))


def resolve_evaluation_model_meta(
    meta: ModelMeta, encode_kwargs: EncodeKwargs
) -> ModelMeta:
    """Resolve cache-relevant metadata before an evaluation starts."""
    resolved_meta = meta
    for applies, resolver in _EVALUATION_META_RESOLVERS:
        if applies(meta):
            resolved_meta = resolver(resolved_meta, encode_kwargs)
    return resolved_meta
