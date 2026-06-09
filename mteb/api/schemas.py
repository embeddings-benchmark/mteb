"""Pydantic response models for the leaderboard API.

Field names mirror ``src/lib/types.ts`` in the SvelteKit frontend; Python keeps
``snake_case`` internally while the JSON serialises as ``camelCase`` via
``alias_generator=to_camel``. Add ``populate_by_name=True`` so the schemas can
still be constructed with Python-style keyword args from adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Union
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

if TYPE_CHECKING:
    from mteb.abstasks.abstask import AbsTask
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.benchmarks._leaderboard_menu import MenuEntry as MtebMenuEntry
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.models.model_meta import ModelMeta

_ModelType = Literal["dense", "cross-encoder", "late-interaction", "sparse", "router"]


def _dedupe_strs(values: list[str]) -> list[str]:
    """Order-preserving unique on a list of strings."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _is_url(value: str) -> bool:
    """True iff ``value`` is an http(s) or data URL (vs. emoji/text icon)."""
    head = value[:7].lower()
    return head.startswith(("http://", "https:/", "data:"))


def _flatten_eval_langs(
    eval_langs: Mapping[str, list[str]] | list[str],
) -> list[str]:
    """Flatten ``eval_langs`` (list or ``Mapping[subset, list]``) into a deduped list."""
    if isinstance(eval_langs, Mapping):
        flat: list[str] = []
        for langs in eval_langs.values():
            flat.extend(langs)
        return _dedupe_strs(flat)
    return list(eval_langs)


def _flatten_task_languages(task: AbsTask | type[AbsTask]) -> list[str]:
    """Flatten ``task.metadata.eval_langs`` into a deduped list."""
    return _flatten_eval_langs(task.metadata.eval_langs)


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
    simplified_task_types: list[str] = []
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
        from mteb.languages import language_label

        languages: set[str] = set()
        task_types: set[str] = set()
        simplified_types: set[str] = set()
        task_names: list[str] = []
        domains: set[str] = set()
        modalities: set[str] = set()
        for task in benchmark.tasks:
            for lang in _flatten_task_languages(task):
                languages.add(language_label(lang))
            task_types.add(str(task.metadata.type))
            simp = getattr(task.metadata, "simplified_task_type", None)
            if simp:
                simplified_types.add(str(simp))
            task_names.append(task.metadata.name)
            for dom in task.metadata.domains or []:
                domains.add(str(dom))
            if task.metadata.modalities:
                for mod in task.metadata.modalities:
                    modalities.add(str(mod))
            else:
                modalities.add("text")
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
            language_view = _dedupe_strs(
                [language_label(c) for c in benchmark.language_view]
            )
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
        from mteb.languages import language_label

        lang_codes = _flatten_eval_langs(metadata.eval_langs)
        labels = _dedupe_strs([language_label(code) for code in lang_codes])
        domains = _dedupe_strs([str(d) for d in (metadata.domains or [])])
        modalities_raw = list(metadata.modalities) if metadata.modalities else ["text"]
        modalities = _dedupe_strs([str(m) for m in modalities_raw])
        try:
            simplified = str(metadata.simplified_task_type)
        except KeyError:
            # Some niche task types aren't in _TASKTYPE2SIMPLIFIEDTASKTYPE.
            simplified = str(metadata.type)
        dataset_path: str | None = None
        try:
            ds = metadata.dataset
            if isinstance(ds, dict):
                p = ds.get("path")
                if p:
                    dataset_path = str(p)
        except Exception:
            dataset_path = None
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
            simplified_type=simplified,
            languages=labels,
            domains=domains,
            modalities=modalities,
            description=metadata.description or "",
            reference=str(metadata.reference) if metadata.reference else None,
            citation=metadata.bibtex_citation or None,
            is_public=bool(getattr(metadata, "is_public", True)),
            source_dataset=dataset_path,
            license=str(metadata.license) if metadata.license else None,
            date_from=date_from,
            date_to=date_to,
            annotations_creators=(
                str(metadata.annotations_creators)
                if metadata.annotations_creators
                else None
            ),
            dialect=list(metadata.dialect) if metadata.dialect else None,
            sample_creation=(
                str(metadata.sample_creation) if metadata.sample_creation else None
            ),
            main_score=str(metadata.main_score) if metadata.main_score else None,
        )


class ModelMetaSchema(_CamelModel):
    """Static model metadata + per-benchmark zero-shot label.

    ``name`` is the canonical ``org/name`` HuggingFace identifier.
    """

    name: str
    url: str | None = None
    zero_shot_pct: int
    active_params_b: float
    total_params_b: float
    embedding_dim: int
    max_tokens: int
    release_date: str | None = None
    model_type: _ModelType
    instruction_tuned: bool
    open_weights: bool
    sentence_transformers_compatible: bool
    modalities: list[str] = ["text"]
    languages: list[str] = []
    citation: str | None = None
    memory_usage_mb: float | None = None
    license: str | None = None
    public_training_code: str | None = None
    # Upstream allows URL or bool; serialised as string either way.
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
        from mteb.benchmarks._create_table import (
            _format_max_tokens,
            _format_n_parameters,
            _get_embedding_size,
        )

        framework = list(meta.framework or [])
        model_type = (meta.model_type or ["dense"])[0]
        n_active = (
            meta.n_active_parameters_override
            if meta.n_active_parameters_override is not None
            else meta.n_parameters
        )
        active_b = _format_n_parameters(n_active)
        total_b = _format_n_parameters(meta.n_parameters)
        embed_dim = _get_embedding_size(meta.embed_dim)
        max_tokens = _format_max_tokens(meta.max_tokens)
        from mteb.languages import language_label as _language_label

        lang_labels = sorted(
            {_language_label(code) for code in (meta.languages or []) if code}
        )
        return cls(
            name=meta.name or "",
            url=str(meta.reference) if meta.reference else None,
            zero_shot_pct=-1 if zero_shot_pct is None else int(zero_shot_pct),
            active_params_b=float(active_b) if active_b is not None else 0.0,
            total_params_b=float(total_b) if total_b is not None else 0.0,
            embedding_dim=int(embed_dim) if embed_dim is not None else 0,
            max_tokens=int(max_tokens) if max_tokens is not None else 0,
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
    active_params_b: float
    total_params_b: float
    embedding_dim: int
    max_tokens: int
    mean_task: float | None
    mean_task_type: float | None
    # Public/Private split means (RTEB/ViDoRe family); null elsewhere.
    mean_public: float | None = None
    mean_private: float | None = None
    scores_by_task_type: dict[str, float]
    scores_by_task: dict[str, float]
    # Tasks the model trained on (drives the frontend's ⚠️ on non-zero-shot scores).
    trained_on_tasks: list[str] = []


class BenchmarkSummarySchema(_CamelModel):
    """Response from ``/benchmarks/{name}/scores``."""

    benchmark_name: str
    task_types: list[str]
    tasks: list[str]
    tasks_meta: list[TaskMetaSchema]
    rows: list[SummaryRowSchema]
    aggregations: list[str] = []


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
    total_params_b: float


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
    # Three-state: True = task is in `model.training_datasets`, False = it
    # isn't, None = the model didn't declare `training_datasets` at all
    # (frontend renders the same NA marker it uses for `zero_shot_pct=-1`).
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


# Frontend discriminates by `'displayName' in item`, so the two coexist in a Union.
MenuChild = Union["MenuEntrySchema", BenchmarkSchema]


class MenuEntrySchema(_CamelModel):
    """Recursive menu entry; children are ``MenuEntrySchema`` or ``BenchmarkSchema``."""

    name: str
    description: str | None = None
    open: bool = False
    children: list[MenuChild] = Field(default_factory=list)

    @classmethod
    def from_menu_entry(cls, entry: MtebMenuEntry) -> MenuEntrySchema:
        """Recursively translate an mteb menu entry tree."""
        from mteb.benchmarks.benchmark import Benchmark

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


MenuEntrySchema.model_rebuild()
