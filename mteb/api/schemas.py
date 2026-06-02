"""Pydantic response models for the leaderboard API.

Field names mirror ``src/lib/types.ts`` in the SvelteKit frontend; Python keeps
``snake_case`` internally while the JSON serialises as ``camelCase`` via
``alias_generator=to_camel``. Add ``populate_by_name=True`` so the schemas can
still be constructed with Python-style keyword args from adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

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
    """Return True iff ``value`` is an http(s) or data URL.

    Used to decide whether a benchmark's ``icon`` should be proxied through
    our cache-friendly /icon route or passed through verbatim (for short
    text/emoji icons like "🌍").
    """
    head = value[:7].lower()
    return head.startswith(("http://", "https:/", "data:"))


def _flatten_task_languages(task: AbsTask | type[AbsTask]) -> list[str]:
    """Flatten the ``eval_langs`` of a task (list or ``dict[subset, list]``) into a deduped list."""
    eval_langs = task.metadata.eval_langs
    if isinstance(eval_langs, dict):
        out: list[str] = []
        seen: set[str] = set()
        for langs in eval_langs.values():
            for lang in langs:
                if lang not in seen:
                    seen.add(lang)
                    out.append(lang)
        return out
    return list(eval_langs)


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
    tasks: list[str]
    domains: list[str]
    modalities: list[str]
    # True when the benchmark is part of the curated leaderboard menu.
    # Off-menu benchmarks (`display_on_leaderboard=False`) are emitted by
    # ``/benchmarks?include_hidden=true`` for the all-benchmarks page.
    display_on_leaderboard: bool = True
    # Names of newer benchmarks that supersede this one (mirrors
    # ``Benchmark.new_version``). Off-menu cards use this to surface
    # "newer version available" instead of just "not on the menu".
    new_version: list[str] | None = None
    # Declarative list of aggregations the leaderboard view should render
    # ("mean_task", "mean_task_type", "task_types", "public_private"). Mirrors
    # ``Benchmark.aggregations`` so the frontend can hide irrelevant columns
    # (e.g. ViDoRe has no per-type breakdown; RTEB has only Mean (Task)).
    aggregations: list[str]

    @classmethod
    def from_benchmark(cls, benchmark: Benchmark) -> BenchmarkSchema:
        """Aggregate per-task metadata up to the benchmark level."""
        from mteb.languages import language_label

        languages: set[str] = set()
        task_types: set[str] = set()
        task_names: list[str] = []
        domains: set[str] = set()
        modalities: set[str] = set()
        for task in benchmark.tasks:
            for lang in _flatten_task_languages(task):
                languages.add(language_label(lang))
            task_types.add(str(task.metadata.type))
            task_names.append(task.metadata.name)
            for dom in task.metadata.domains or []:
                domains.add(str(dom))
            if task.metadata.modalities:
                for mod in task.metadata.modalities:
                    modalities.add(str(mod))
            else:
                modalities.add("text")
        # The icon field is intentionally polymorphic: it can be a URL (flag
        # SVG, hosted PNG) or short text/emoji ("🌍"). URLs get rewritten to
        # our cache-friendly proxy; everything else is passed through
        # verbatim so the frontend can render it as text instead of <img>.
        from urllib.parse import quote

        icon_value: str | None
        if benchmark.icon and _is_url(benchmark.icon):
            icon_value = f"/icon/{quote(benchmark.name, safe='')}"
        else:
            icon_value = benchmark.icon or None
        return cls(
            name=benchmark.name,
            display_name=benchmark.display_name or benchmark.name,
            icon=icon_value,
            description=benchmark.description or "",
            reference=str(benchmark.reference) if benchmark.reference else None,
            citation=benchmark.citation,
            languages=sorted(languages),
            task_types=sorted(task_types),
            tasks=task_names,
            domains=sorted(domains),
            modalities=sorted(modalities),
            display_on_leaderboard=bool(benchmark.display_on_leaderboard),
            new_version=list(benchmark.new_version) if benchmark.new_version else None,
            aggregations=[a.value for a in benchmark.aggregations],
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
    # Whether the task's dataset is openly published. Frontend uses this to
    # recompute Mean (Public) / Mean (Private) when the user filters tasks on
    # benchmarks that surface the public/private split (ViDoRe family).
    is_public: bool = True
    # Extended dataset metadata for the task detail card.
    source_dataset: str | None = None
    license: str | None = None
    # Date range the source data was collected over, formatted as ISO strings.
    # ``None`` when the upstream `date` tuple is unset.
    date_from: str | None = None
    date_to: str | None = None
    annotations_creators: str | None = None
    dialect: list[str] | None = None
    sample_creation: str | None = None

    @classmethod
    def from_task_metadata(cls, metadata: TaskMetadata) -> TaskMetaSchema:
        """Build the API schema view of a :class:`TaskMetadata` (no instantiation needed)."""
        from mteb.languages import language_label

        if isinstance(metadata.eval_langs, dict):
            seen_codes: set[str] = set()
            lang_codes: list[str] = []
            for ls in metadata.eval_langs.values():
                for code in ls:
                    if code not in seen_codes:
                        seen_codes.add(code)
                        lang_codes.append(code)
        else:
            lang_codes = list(metadata.eval_langs)

        labels = _dedupe_strs([language_label(code) for code in lang_codes])
        domains = _dedupe_strs([str(d) for d in (metadata.domains or [])])
        modalities_raw = list(metadata.modalities) if metadata.modalities else ["text"]
        modalities = _dedupe_strs([str(m) for m in modalities_raw])
        try:
            simplified = str(metadata.simplified_task_type)
        except KeyError:
            # Some niche task types aren't in ``_TASKTYPE2SIMPLIFIEDTASKTYPE``;
            # fall back to the raw type rather than 500-ing the whole tasks list.
            simplified = str(metadata.type)
        # Source dataset path: prefer the loader's ``dataset["path"]`` (HF
        # repo id, etc.); fall back to ``None`` when unset.
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
        )


class ModelMetaSchema(_CamelModel):
    """Static model metadata + per-benchmark zero-shot label.

    Note: ``name`` is the canonical HuggingFace-style identifier (``org/name``).
    Splitting it into org and display name is left to the client so the API
    isn't in the business of deciding what's a "display name".
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
    citation: str | None = None
    # Extended metadata surfaced on the model detail card. All optional —
    # missing values render as "—" on the frontend rather than blocking.
    memory_usage_mb: float | None = None
    license: str | None = None
    public_training_code: str | None = None
    # Upstream allows either a URL or a bool ("yes/no, no link"); we serialise
    # both as a string so the frontend can render uniformly.
    public_training_data: str | None = None
    adapted_from: str | None = None
    superseded_by: str | None = None
    # Optional pip extras users need to install for this model
    # (e.g. ["api", "vision"]).
    extra_requirements_groups: list[str] | None = None
    # Dataset names this model was trained on. Entries that exist in mteb's
    # task registry can be linked to /tasks/<name>; others render as plain
    # text on the frontend.
    training_datasets: list[str] | None = None

    @classmethod
    def from_model_meta(
        cls, meta: ModelMeta, *, zero_shot_pct: int | None = None
    ) -> ModelMetaSchema:
        """Build the API schema view of a :class:`ModelMeta`.

        ``zero_shot_pct`` is the only per-call varying field; callers that need
        to swap it on a pre-built base instance should use
        ``schema.model_copy(update={"zero_shot_pct": pct})`` (the
        :mod:`mteb.api.adapters` cache does this).
        """
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
    # Public/Private split means, when the benchmark separates evaluation
    # tasks into a published subset and a held-out subset (RTEB/ViDoRe
    # family). Most benchmarks have neither — both are null and the frontend
    # omits the corresponding columns.
    mean_public: float | None = None
    mean_private: float | None = None
    scores_by_task_type: dict[str, float]
    scores_by_task: dict[str, float]


class BenchmarkSummarySchema(_CamelModel):
    """Response from ``/benchmarks/{name}/summary``."""

    benchmark_name: str
    task_types: list[str]
    tasks: list[str]
    tasks_meta: list[TaskMetaSchema]
    rows: list[SummaryRowSchema]
    # Declarative list mirroring ``Benchmark.aggregations`` so the frontend
    # can render only the columns this benchmark actually surfaces.
    aggregations: list[str] = []


class TaskScoreRowSchema(_CamelModel):
    """One row of `/tasks/{name}/scores` — per-model score on a single task.

    ``subset_scores`` maps the HuggingFace subset id (often a language pair or
    split label) to that model's main-metric value for that subset, taken
    straight from the result's ``scores`` dict. ``score`` is the mean across
    *all* of the task's subsets — set to ``null`` when the model is missing
    any subset, since a partial mean isn't comparable to a full one.
    """

    rank: int
    model: ModelMetaSchema
    score: float | None
    subset_scores: dict[str, float]
    benchmarks: list[str]


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


# Recursive menu type. The frontend discriminates by `'displayName' in item`, so
# BenchmarkSchema (which has `displayName`) and MenuEntrySchema (which doesn't)
# coexist in a Union.
MenuChild = Annotated[
    Union["MenuEntrySchema", BenchmarkSchema],
    Field(discriminator=None),
]


class MenuEntrySchema(_CamelModel):
    """Recursive menu entry — children may be ``MenuEntrySchema`` or ``BenchmarkSchema``."""

    name: str
    description: str | None = None
    open: bool = False
    children: list[MenuChild] = Field(default_factory=list)

    @classmethod
    def from_menu_entry(cls, entry: MtebMenuEntry) -> MenuEntrySchema:
        """Recursively translate an mteb menu entry tree into its API schema."""
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
