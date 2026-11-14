from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, get_args

from mteb.types import Modalities, PromptType

from .models_protocols import CrossEncoderProtocol, EncoderProtocol, SearchProtocol
from .search_wrappers import SearchCrossEncoderWrapper, SearchEncoderWrapper

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

    from .models_protocols import MTEBModels

_MODALITIES = frozenset(get_args(Modalities))
_TEXT_COLUMNS = frozenset(
    {"text", "title", "body", "query", "instruction", "conversation"}
)


def _normalize_modalities(
    value: Sequence[Modalities] | None, name: str
) -> tuple[Modalities, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonempty sequence of modality names.")
    if any(not isinstance(item, str) or item not in _MODALITIES for item in value):
        raise ValueError(f"{name} must contain only {sorted(_MODALITIES)}.")
    return tuple(sorted(set(value)))


class ModalityFilterWrapper:
    """Select query/corpus inputs for a retrieval or reranking experiment.

    The wrapper exposes SearchProtocol, adapting encoders and cross-encoders as
    needed. Custom search implementations retain their own index/search methods
    and must honor the supplied datasets. This does not configure external indexes.

    Examples:
        >>> import mteb
        >>> model = mteb.get_model("nvidia/Nemotron-3.5-Embed-1B-BF16")
        >>> selected = ModalityFilterWrapper(model, corpus_modalities=["image"])
        >>> mteb.evaluate(selected, mteb.get_task("Vidore3HrRetrieval.v2"))
    """

    def __init__(
        self,
        model: MTEBModels,
        *,
        query_modalities: Sequence[Modalities] | None = None,
        corpus_modalities: Sequence[Modalities] | None = None,
    ) -> None:
        """Configure exact, nonempty selections; None leaves a side unchanged.

        Built-in embedding caches inside the wrapped search are supported. An
        embedding cache outside this search-only wrapper, nested modality filters,
        and standard encoder wrappers with a reusable index are not supported.
        Use separate instances for concurrent evaluations of stateful models.
        """
        if isinstance(model, ModalityFilterWrapper):
            raise ValueError(
                "Nested ModalityFilterWrapper instances are not supported."
            )
        self._query_modalities = _normalize_modalities(
            query_modalities, "query_modalities"
        )
        self._corpus_modalities = _normalize_modalities(
            corpus_modalities, "corpus_modalities"
        )
        self._configuration = {
            name: list(value)
            for name, value in (
                ("query_modalities", self._query_modalities),
                ("corpus_modalities", self._corpus_modalities),
            )
            if value is not None
        }
        meta = model.mteb_model_meta
        for name, selected in self._configuration.items():
            if meta.modalities and (missing := set(selected) - set(meta.modalities)):
                raise ValueError(
                    f"{name} requests unsupported model modalities: {sorted(missing)}."
                )
        self.mteb_model_meta = meta.model_copy(
            update={
                "experiment_kwargs": {
                    **(meta.experiment_kwargs or {}),
                    **self._configuration,
                }
                or None
            },
            deep=True,
        )
        # Prioritize custom search even if it also exposes encode or predict.
        if isinstance(model, SearchProtocol):
            self.model: SearchProtocol = model
        elif isinstance(model, CrossEncoderProtocol):
            self.model = SearchCrossEncoderWrapper(model)
        elif isinstance(model, EncoderProtocol):
            self.model = SearchEncoderWrapper(model)
        else:
            raise TypeError("Expected a search model, encoder, or cross-encoder.")
        if (
            self._configuration
            and isinstance(self.model, SearchEncoderWrapper)
            and self.model.index_backend is not None
        ):
            raise ValueError(
                "ModalityFilterWrapper does not support SearchEncoderWrapper with "
                "index_backend: its index may append embeddings from other experiments. "
                "Use an encoder without an index backend, or a custom SearchProtocol "
                "that rebuilds its index from the supplied corpus."
            )

    def _effective_metadata(self, metadata: TaskMetadata) -> TaskMetadata:
        if metadata._modality_filter_configuration is not None:
            raise ValueError(
                "Nested modality filters inside custom search are not supported."
            )
        if not self._configuration:
            return metadata
        sides = []
        for role, selected in (
            (PromptType.query, self._query_modalities),
            (PromptType.document, self._corpus_modalities),
        ):
            available = metadata.get_modalities(role)
            if selected is not None:
                missing = set(selected) - set(available)
                if missing:
                    raise ValueError(
                        f"Task {metadata.name} has no {sorted(missing)} on its {role.value} side."
                    )
            # Preserve the task's category ordering for retained modalities.
            sides.append([m for m in available if selected is None or m in selected])
        category = "2".join("".join(m[0] for m in side) for side in sides)
        effective = metadata.model_copy(
            update={
                "category": category,
                "modalities": sorted(set(sides[0] + sides[1])),
            },
            deep=True,
        )
        effective._modality_filter_configuration = {
            key: list(value) for key, value in self._configuration.items()
        }
        return effective

    @staticmethod
    def _select(
        dataset: Dataset, selected: tuple[Modalities, ...] | None, side: str
    ) -> Dataset:
        if selected is None:
            return dataset
        missing = set(selected) - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"The {side} dataset is missing requested modalities: {sorted(missing)}."
            )
        excluded = _MODALITIES - set(selected)
        if "text" in excluded:
            excluded |= _TEXT_COLUMNS
        return dataset.remove_columns(sorted(excluded & set(dataset.column_names)))

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Delegate indexing with only the selected corpus content."""
        metadata = self._effective_metadata(task_metadata)
        filtered = self._select(corpus, self._corpus_modalities, "corpus")
        self.model.index(
            cast("CorpusDatasetType", filtered),
            task_metadata=metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
            **kwargs,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> RetrievalOutputType:
        """Delegate search with selected queries and unchanged candidate/options."""
        metadata = self._effective_metadata(task_metadata)
        filtered = self._select(queries, self._query_modalities, "query")
        return self.model.search(
            cast("QueryDatasetType", filtered),
            task_metadata=metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
            **kwargs,
        )
