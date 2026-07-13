"""Pydantic response models for the leaderboard API.

Field names mirror ``src/lib/types.ts`` in the SvelteKit frontend; JSON
serialises as ``camelCase`` via ``alias_generator=to_camel``.
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
from mteb.models.model_meta import MODEL_TYPES
from mteb.types import Modalities

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.benchmarks._leaderboard_menu import MenuEntry
    from mteb.models.model_meta import ModelMeta


def _is_url(value: str) -> bool:
    """True iff ``value`` is an http(s) or data URL."""
    head = value[:7].lower()
    return head.startswith(("http://", "https:/", "data:"))


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        protected_namespaces=(),
    )


class BenchmarkSchema(_CamelModel):
    """Top-level benchmark metadata."""

    name: str
    display_name: str
    icon: str | None = None
    description: str
    reference: str | None = None
    citation: str | None = None
    languages: list[str]
    task_types: list[str]
    # Drives the /benchmarks "Task group" filter facet.
    simplified_task_types: list[str] = Field(default_factory=list)
    tasks: list[str]
    domains: list[str]
    modalities: list[Modalities]
    display_on_leaderboard: bool = True
    new_version: list[str] | None = None
    aggregations: list[str]
    # False on benchmarks whose tasks aren't tracked in training-data annotations (e.g. ViDoRe).
    show_zero_shot: bool = True
    # Distinct models evaluated on every task in this benchmark.
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
        # URLs go through the /icon proxy; emoji/text pass through verbatim.
        icon_value: str | None
        if benchmark.icon and _is_url(benchmark.icon):
            icon_value = f"/v1/icon/{quote(benchmark.name, safe='')}"
        else:
            icon_value = benchmark.icon or None
        language_view: list[str] | Literal["all"] | None
        if benchmark.language_view == "all":
            language_view = "all"
        elif benchmark.language_view:
            language_view = sorted({language_label(c) for c in benchmark.language_view})
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
            new_version=list(benchmark.superseded_by)
            if benchmark.superseded_by
            else None,
            aggregations=[a.value for a in benchmark.aggregations],
            show_zero_shot=bool(benchmark.show_zero_shot),
            language_view=language_view,
        )


class TaskMetaSchema(_CamelModel):
    """Static task metadata."""

    name: str
    type: str
    simplified_type: str
    languages: list[str]
    domains: list[str] | None
    modalities: list[Modalities]
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
    # Distinct models evaluated on every (subset, split) cell the task declares.
    # Partial-coverage models excluded so the count doesn't overstate completeness.
    num_models: int = 0

    @classmethod
    def from_task_metadata(cls, metadata: TaskMetadata) -> TaskMetaSchema:
        """Build the API schema view of a `TaskMetadata`."""
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


class Co2EstimateSchema(_CamelModel):
    """Estimated inference carbon cost with the assumptions behind it (for a badge + popover).

    All figures are approximate and reflect benchmark conditions, not real-world deployment.
    """

    g_co2_per_million_tokens: float
    active_parameters: int | None
    benchmark_hardware: str
    carbon_intensity_g_per_kwh: float
    pue: float


class ModelMetaSchema(_CamelModel):
    """Static model metadata; ``name`` is the canonical ``org/name`` HF identifier."""

    name: str
    url: str | None = None
    zero_shot_pct: int
    active_params_b: float | None
    total_params_b: float | None
    embedding_dim: int | None
    max_tokens: float | None
    release_date: str | None = None
    model_type: MODEL_TYPES
    instruction_tuned: bool
    open_weights: bool
    openness: dict[str, bool] = Field(default_factory=dict)
    openness_score: int = 0
    sentence_transformers_compatible: bool
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    languages: list[str] = Field(default_factory=list)
    citation: str | None = None
    memory_usage_mb: float | None = None
    co2_estimate: Co2EstimateSchema | None = None
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
        """Build the API schema view of a `ModelMeta`."""
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
            openness=meta.openness,
            openness_score=meta.openness_score,
            sentence_transformers_compatible="Sentence Transformers" in framework,
            modalities=[str(m) for m in (meta.modalities or ["text"])],
            languages=lang_labels,
            citation=meta.citation or None,
            memory_usage_mb=float(meta.memory_usage_mb)
            if meta.memory_usage_mb is not None
            else None,
            co2_estimate=(
                Co2EstimateSchema(**meta.co2_cost_estimate)
                if meta.co2_cost_estimate is not None
                else None
            ),
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
    """One row of a benchmark summary — a model with its aggregate scores."""

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
    """Response from /benchmarks/{name}/scores"""

    benchmark_name: str
    task_types: list[str]
    tasks: list[str]
    tasks_meta: list[TaskMetaSchema]
    rows: list[SummaryRowSchema]
    aggregations: list[str] = Field(default_factory=list)
    show_zero_shot: bool = True


class BenchmarkPerLanguageRowSchema(_CamelModel):
    """One model's per-language scores; keys are human labels (``"English"``)."""

    model_name: str
    scores_by_language: dict[str, float]


class BenchmarkPerLanguageSchema(_CamelModel):
    """Response from ``/v1/benchmarks/{name}/per-language``; lazy-loaded by the tab."""

    benchmark_name: str
    rows: list[BenchmarkPerLanguageRowSchema]


class LeaderModelSchema(_CamelModel):
    """Slim model identity for the leaders endpoint — just name + type for home-tile rendering."""

    name: str
    model_type: MODEL_TYPES


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

    ``score`` is mean across subsets (each subset = max across splits the model
    ran), or ``null`` for partial coverage. ``subset_scores[subset][split]``
    lets the frontend pivot either way without a second request.
    """

    rank: int
    model: ModelMetaSchema
    score: float | None
    subset_scores: dict[str, dict[str, float]]
    benchmarks: list[str]
    trained_on: bool | None = None


class TaskScoresSchema(_CamelModel):
    """Response from `/tasks/{name}/scores`."""

    task: TaskMetaSchema
    benchmarks: list[str]
    subsets: list[str]
    splits: list[str]
    rows: list[TaskScoreRowSchema]


class ModelScoreRowSchema(_CamelModel):
    """Per-benchmark score for one model."""

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
    def from_menu_entry(cls, entry: MenuEntry) -> MenuEntrySchema:
        """Translate an mteb menu entry tree."""
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
