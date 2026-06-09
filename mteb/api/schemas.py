"""Pydantic response models for the leaderboard API.

Field names mirror ``src/lib/types.ts`` in the SvelteKit frontend; Python keeps
``snake_case`` internally while the JSON serialises as ``camelCase`` via
``alias_generator=to_camel``. Add ``populate_by_name=True`` so the schemas can
still be constructed with Python-style keyword args from adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from mteb.benchmarks._create_table import (
    _format_max_tokens,
    _format_n_parameters,
    _get_embedding_size,
)
from mteb.benchmarks.benchmark import Benchmark
from mteb.languages import language_label

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.benchmarks._leaderboard_menu import MenuEntry as MtebMenuEntry
    from mteb.models.model_meta import ModelMeta

_ModelType = Literal["dense", "cross-encoder", "late-interaction", "sparse", "router"]


def _is_url(value: str) -> bool:
    """True iff ``value`` is an http(s) or data URL (vs. emoji/text icon)."""
    head = value[:7].lower()
    return head.startswith(("http://", "https:/", "data:"))


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        protected_namespaces=(),
    )


class BenchmarkSchema(_CamelModel):
    """Top-level benchmark metadata returned by ``/benchmarks`` and ``/benchmarks/{name}``."""

    name: str
    display_name: str
    icon: str | None = None
    description: str
    reference: str | None = None
    citation: str | None = None
    languages: list[str]
    task_types: list[str]
    # Borda-compatible buckets (retrieval, classification, …). Drives the
    # /benchmarks "Task group" filter facet.
    simplified_task_types: list[str] = Field(default_factory=list)
    tasks: list[str]
    domains: list[str]
    modalities: list[str]
    display_on_leaderboard: bool = True
    new_version: list[str] | None = None
    # Mirrors ``Benchmark.aggregations`` so the frontend hides columns the
    # benchmark doesn't produce.
    aggregations: list[str]
    # Distinct models with at least one score on this benchmark.
    num_models: int = 0
    # ``"all"`` = every language in the data; ``None`` = no per-language view.
    language_view: list[str] | Literal["all"] | None = None

    @classmethod
    def from_benchmark(cls, benchmark: Benchmark) -> BenchmarkSchema:
        """Aggregate per-task metadata up to the benchmark level."""
        languages: set[str] = set()
        task_types: set[str] = set()
        simplified_types: set[str] = set()
        task_names: list[str] = []
        domains: set[str] = set()
        modalities: set[str] = set()
        for task in benchmark.tasks:
            for lang in task.languages:
                languages.add(language_label(lang))
            task_types.add(task.metadata.type)
            simplified_types.add(task.metadata.simplified_task_type)
            task_names.append(task.metadata.name)
            for dom in task.metadata.domains or []:
                domains.add(str(dom))
            for mod in task.metadata.modalities:
                modalities.add(str(mod))
        # icon is polymorphic — URLs go through the /icon proxy; emoji/text
        # are passed verbatim.
        icon_value: str | None
        if benchmark.icon and _is_url(benchmark.icon):
            icon_value = f"/v1/icon/{quote(benchmark.name, safe='')}"
        else:
            icon_value = benchmark.icon or None
        language_view: list[str] | Literal["all"] | None
        if benchmark.language_view == "all":
            language_view = "all"
        elif benchmark.language_view:
            language_view = sorted([language_label(c) for c in benchmark.language_view])
        else:
            language_view = None
        return cls(
            name=benchmark.name,
            display_name=benchmark.display_name or benchmark.name,
            icon=icon_value,
            description=benchmark.description or "",
            reference=str(benchmark.reference) if benchmark.reference else None,
            citation=benchmark.citation,
            languages=sorted(languages),
            task_types=sorted(task_types),
            simplified_task_types=sorted(simplified_types),
            tasks=task_names,
            domains=sorted(domains),
            modalities=sorted(modalities),
            display_on_leaderboard=bool(benchmark.display_on_leaderboard),
            new_version=list(benchmark.new_version) if benchmark.new_version else None,
            aggregations=[a.value for a in benchmark.aggregations],
            language_view=language_view,
        )


class TaskMetaSchema(_CamelModel):
    """Static task metadata (returned by ``/tasks`` and embedded in summary payloads)."""

    name: str
    type: str
    simplified_type: str
    languages: list[str]
    domains: list[str]
    modalities: list[str]
    description: str
    reference: str | None = None
    citation: str | None = None
    # Recompute target for Mean (Public) / Mean (Private) on ViDoRe-family benches.
    is_public: bool = True
    source_dataset: str | None = None
    license: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    annotations_creators: str | None = None
    dialect: list[str] | None = None
    sample_creation: str | None = None
    main_score: str | None = None
    num_models: int = 0

    @classmethod
    def from_task_metadata(cls, metadata: TaskMetadata) -> TaskMetaSchema:
        """Build the API schema view of a :class:`TaskMetadata`."""
        labels = [language_label(code) for code in metadata.languages]
        date_from: str | None = None
        date_to: str | None = None
        if metadata.date:
            try:
                date_from = str(metadata.date[0])
                date_to = str(metadata.date[1])
            except (IndexError, TypeError):
                pass
        return cls(
            name=metadata.name,
            type=str(metadata.type),
            simplified_type=metadata.simplified_task_type,
            languages=labels,
            domains=metadata.domains or None,
            modalities=metadata.modalities,
            description=metadata.description,
            reference=metadata.reference,
            citation=metadata.bibtex_citation,
            is_public=metadata.is_public,
            source_dataset=metadata.dataset["path"],
            license=metadata.license,
            date_from=date_from,
            date_to=date_to,
            annotations_creators=metadata.annotations_creators,
            dialect=metadata.dialect,
            sample_creation=metadata.sample_creation,
            main_score=metadata.main_score,
        )


class ModelMetaSchema(_CamelModel):
    """Static model metadata + per-benchmark zero-shot label.

    ``name`` is the canonical ``org/name`` HuggingFace identifier.
    """

    name: str
    url: str | None = None
    zero_shot_pct: int
    active_params_b: float | None
    total_params_b: float | None
    embedding_dim: int | None
    max_tokens: float | None
    release_date: str | None = None
    model_type: _ModelType
    instruction_tuned: bool
    open_weights: bool
    sentence_transformers_compatible: bool
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    languages: list[str] = Field(default_factory=list)
    citation: str | None = None
    memory_usage_mb: float | None = None
    license: str | None = None
    public_training_code: str | None = None
    public_training_data: str | None = None
    adapted_from: str | None = None
    superseded_by: str | None = None
    extra_requirements_groups: list[str] | None = None
    training_datasets: list[str] | None = None

    @classmethod
    def from_model_meta(
        cls, meta: ModelMeta, *, zero_shot_pct: int | None = None
    ) -> ModelMetaSchema:
        """Build the API schema view of a :class:`ModelMeta`."""
        framework = list(meta.framework or [])
        model_type = (meta.model_type or ["dense"])[0]
        n_active = (
            meta.n_active_parameters_override
            if meta.n_active_parameters_override is not None
            else meta.n_parameters
        )

        lang_labels = sorted(
            {language_label(code) for code in (meta.languages or []) if code}
        )
        return cls(
            name=meta.name or "",
            url=meta.reference,
            zero_shot_pct=-1 if zero_shot_pct is None else int(zero_shot_pct),
            active_params_b=_format_n_parameters(n_active),
            total_params_b=_format_n_parameters(meta.n_parameters),
            embedding_dim=_get_embedding_size(meta.embed_dim),
            max_tokens=_format_max_tokens(meta.max_tokens),
            release_date=str(meta.release_date) if meta.release_date else None,
            model_type=model_type,
            instruction_tuned=bool(meta.use_instructions)
            if meta.use_instructions is not None
            else False,
            open_weights=bool(meta.open_weights)
            if meta.open_weights is not None
            else False,
            sentence_transformers_compatible="Sentence Transformers" in framework,
            modalities=[str(m) for m in (meta.modalities or ["text"])],
            languages=lang_labels,
            citation=meta.citation or None,
            memory_usage_mb=float(meta.memory_usage_mb)
            if meta.memory_usage_mb is not None
            else None,
            license=str(meta.license) if meta.license else None,
            public_training_code=str(meta.public_training_code)
            if meta.public_training_code
            else None,
            public_training_data=(
                None
                if meta.public_training_data is None
                else str(meta.public_training_data)
            ),
            adapted_from=meta.adapted_from or None,
            superseded_by=meta.superseded_by or None,
            extra_requirements_groups=(
                list(meta.extra_requirements_groups)
                if meta.extra_requirements_groups
                else None
            ),
            training_datasets=(
                sorted(meta.training_datasets) if meta.training_datasets else None
            ),
        )


class SummaryRowSchema(_CamelModel):
    """One row of a benchmark summary table — a model with its aggregate scores."""

    rank: int
    model: ModelMetaSchema
    zero_shot_pct: int
    active_params_b: float | None
    total_params_b: float | None
    embedding_dim: int | None
    max_tokens: int | None
    mean_task: float | None
    mean_task_type: float | None
    mean_public: float | None = None
    mean_private: float | None = None
    scores_by_task_type: dict[str, float]
    scores_by_task: dict[str, float]
    trained_on_tasks: list[str] = Field(default_factory=list)


class BenchmarkSummarySchema(_CamelModel):
    """Response from ``/benchmarks/{name}/scores``."""

    benchmark_name: str
    task_types: list[str]
    tasks: list[str]
    tasks_meta: list[TaskMetaSchema]
    rows: list[SummaryRowSchema]
    aggregations: list[str] = Field(default_factory=list)


class BenchmarkPerLanguageRowSchema(_CamelModel):
    """One model's per-language scores; keys are human labels (``"English"``)."""

    model_name: str
    scores_by_language: dict[str, float]


class BenchmarkPerLanguageSchema(_CamelModel):
    """Response from ``/v1/benchmarks/{name}/per-language``.

    Replaces the synthetic placeholder PerLanguageTab used to render.
    Lazily fetched by the tab on mount so the per-language aggregate
    (long-frame explode + group_by) only runs when a user opens it.
    """

    benchmark_name: str
    rows: list[BenchmarkPerLanguageRowSchema]


class LeaderModelSchema(_CamelModel):
    """Slim model identity for the leaders endpoint.

    Returned per-bucket from ``/benchmarks/{name}/leaders`` — just the
    fields the home page needs to render a one-line leaderboard entry
    with the right model-type tint and an internal link to
    ``/models/[name]``. Strip everything else (release date, max tokens,
    embedding dim, …) so the payload is tiny vs. ``/scores``'s full
    ``ModelMetaSchema``. ``name`` is the canonical ``org/name``
    HuggingFace identifier; the frontend splits on ``/`` when it needs
    a display-only name.
    """

    name: str
    model_type: _ModelType


class LeaderRowSchema(_CamelModel):
    """The highest-meanTask model in one size bucket."""

    rank: int
    model: LeaderModelSchema
    mean_task: float | None = None
    total_params_b: float | None = None


class BucketLeaderSchema(_CamelModel):
    """One size-bucket result; ``max=None`` means ``+inf``, ``leader=None`` means empty."""

    min: float
    max: float | None = None
    leader: LeaderRowSchema | None = None


class BenchmarkLeadersSchema(_CamelModel):
    """Response from ``/benchmarks/{name}/leaders``; bucket order matches request."""

    benchmark_name: str
    buckets: list[BucketLeaderSchema]


class TaskScoreRowSchema(_CamelModel):
    """One row of `/tasks/{name}/scores`.

    ``score`` is the mean across every subset the task offers, or ``null`` when
    the model is missing any subset (partial means aren't comparable).
    """

    rank: int
    model: ModelMetaSchema
    score: float | None
    subset_scores: dict[str, float]
    benchmarks: list[str]
    trained_on: bool | None = None


class TaskScoresSchema(_CamelModel):
    """Response from `/tasks/{name}/scores`."""

    task: TaskMetaSchema
    benchmarks: list[str]
    subsets: list[str]
    rows: list[TaskScoreRowSchema]


class ModelScoreRowSchema(_CamelModel):
    """One row of `/models/{name}/scores` — per-benchmark score for a single model."""

    benchmark_name: str
    benchmark_display_name: str
    rank: int
    total_models: int
    mean_task: float | None
    mean_task_type: float | None
    zero_shot_pct: int
    task_types: list[str]
    scores_by_task_type: dict[str, float]


class ModelScoresSchema(_CamelModel):
    """Response from `/models/{name}/scores`."""

    model: ModelMetaSchema
    rows: list[ModelScoreRowSchema]


class MenuEntrySchema(_CamelModel):
    """Recursive menu entry; children are ``MenuEntrySchema`` or ``BenchmarkSchema``."""

    name: str
    description: str | None = None
    open: bool = False
    children: list[MenuEntrySchema | BenchmarkSchema] = Field(default_factory=list)

    @classmethod
    def from_menu_entry(cls, entry: MtebMenuEntry) -> MenuEntrySchema:
        """Recursively translate an mteb menu entry tree."""
        children: list[BenchmarkSchema | MenuEntrySchema] = []
        for child in entry.benchmarks:
            if isinstance(child, Benchmark):
                children.append(BenchmarkSchema.from_benchmark(child))
            else:
                children.append(cls.from_menu_entry(child))
        return cls(
            name=entry.name or "",
            description=entry.description,
            open=entry.open,
            children=children,
        )
