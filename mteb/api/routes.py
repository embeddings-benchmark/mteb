"""FastAPI routes for the leaderboard.

Handlers return pydantic schemas (or lists of them) directly — FastAPI
serialises them via pydantic-core (Rust) on the way out. ETag-based 304
revalidation is handled by :class:`mteb.api.app.ETagMiddleware`, so per-route
code stays focused on producing the schema.

Caches keep pydantic instances (not bytes); we trade a single-digit-ms
``model_dump_json`` per warm request for a uniformly typed cache that's
easier to slice and reuse.
"""

from __future__ import annotations

import functools
import logging
from importlib.resources import files
from typing import Annotated

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Response

from mteb.api.adapters import (
    benchmark_to_schema,
    menus_to_schemas,
    model_meta_to_schema,
    task_to_meta_schema,
)
from mteb.api.cache import (
    get_model_scores,
    get_summary,
    get_task_scores,
)
from mteb.api.icons import get_icon
from mteb.api.metrics import (
    BENCHMARK_SELECTIONS,
    MODEL_SELECTIONS,
    TASK_SELECTIONS,
    render_metrics,
)
from mteb.api.schemas import (  # noqa: TC001 — FastAPI inspects return annotations at registration
    BenchmarkSchema,
    BenchmarkSummarySchema,
    MenuEntrySchema,
    ModelMetaSchema,
    ModelScoresSchema,
    TaskMetaSchema,
    TaskScoresSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@functools.lru_cache(maxsize=1)
def _num_models_map() -> dict[str, int]:
    """{benchmark_name -> distinct model count}.

    Cheap: reads the in-memory per-benchmark polars frames, computes a
    distinct-count on ``model_name``, returns a dict. Cached for the
    process lifetime so every BenchmarkSchema served by the API — list,
    detail, or menu — can carry a real ``num_models`` without repeating
    the polars work per request.
    """
    from mteb.api.cache import _load_per_benchmark_frames

    try:
        frames, _ = _load_per_benchmark_frames()
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for name, frame in frames.items():
        if "model_name" in frame.columns:
            counts[name] = int(frame["model_name"].n_unique())
    return counts


def _with_num_models(schema: BenchmarkSchema) -> BenchmarkSchema:
    """Overlay the cached ``num_models`` count onto a ``BenchmarkSchema``.

    Never mutates the input — schemas are memoised in
    ``adapters._benchmark_schema_cache`` and shared across requests.
    """
    n = _num_models_map().get(schema.name, 0)
    if n == schema.num_models:
        return schema
    return schema.model_copy(update={"num_models": n})


@functools.lru_cache(maxsize=1)
def _task_num_models_map() -> dict[str, int]:
    """{task_name -> distinct model count}.

    Cheap: reads the in-memory unified results frame, groups by
    ``task_name`` and counts distinct ``model_name``. Cached for the
    process lifetime so every ``TaskMetaSchema`` returned by ``/tasks``
    or ``/tasks/{name}`` can carry a real ``num_models`` without a
    polars roundtrip per task.
    """
    from mteb.api.cache import _load_per_benchmark_frames

    try:
        _, unified = _load_per_benchmark_frames()
    except Exception:
        return {}
    if unified.is_empty() or "task_name" not in unified.columns:
        return {}
    grouped = (
        unified.lazy()
        .group_by("task_name")
        .agg(pl.col("model_name").n_unique().alias("n"))
        .collect()
    )
    return {row["task_name"]: int(row["n"]) for row in grouped.to_dicts()}


def _with_task_num_models(schema: TaskMetaSchema) -> TaskMetaSchema:
    """Overlay the cached ``num_models`` count onto a ``TaskMetaSchema``."""
    n = _task_num_models_map().get(schema.name, 0)
    if n == schema.num_models:
        return schema
    return schema.model_copy(update={"num_models": n})


def _patch_menu_counts(entries: list[MenuEntrySchema]) -> list[MenuEntrySchema]:
    """Walk the menu tree and overlay ``num_models`` on every benchmark child."""
    patched: list[MenuEntrySchema] = []
    for entry in entries:
        new_children: list[BenchmarkSchema | MenuEntrySchema] = []
        for child in entry.children:
            if isinstance(child, BenchmarkSchema):
                new_children.append(_with_num_models(child))
            else:
                new_children.extend(_patch_menu_counts([child]))
        patched.append(entry.model_copy(update={"children": new_children}))
    return patched


@functools.lru_cache(maxsize=1)
def _menu_schemas() -> list[MenuEntrySchema]:
    from mteb.benchmarks._leaderboard_menu import (
        GP_BENCHMARK_ENTRIES,
        R_BENCHMARK_ENTRIES,
    )

    raw = menus_to_schemas(list(GP_BENCHMARK_ENTRIES) + list(R_BENCHMARK_ENTRIES))
    return _patch_menu_counts(raw)


@functools.lru_cache(maxsize=2)
def _benchmark_schemas(include_hidden: bool = False) -> list[BenchmarkSchema]:
    import mteb

    if include_hidden:
        # Every registered benchmark — including ones not on the curated
        # leaderboard menu (display_on_leaderboard=False). Used by the
        # all-benchmarks page so off-menu benchmarks are still discoverable.
        benches = mteb.get_benchmarks()
    else:
        benches = mteb.get_benchmarks(display_on_leaderboard=True)
    return [_with_num_models(benchmark_to_schema(b)) for b in benches]


@router.get("/health")
async def health() -> dict[str, bool]:
    """Liveness probe — returns ``{"ok": True}`` once the app is up."""
    return {"ok": True}


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint — exposes per-route counters and latency histograms."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


# Crawlers (and the Spaces health checker) probe this on every cold start.
# Serving a tiny disallow-everything body keeps the log clean and tells
# search engines not to index the JSON API. Cached aggressively because
# the response never changes.
_ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


@router.get("/robots.txt", include_in_schema=False)
async def robots_txt() -> Response:
    """Cached `Disallow: /` body so crawler probes stop 404-ing in the log."""
    return Response(
        content=_ROBOTS_TXT,
        media_type="text/plain",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# Shipped as package data via pyproject's ``[tool.setuptools.package-data]``
# so the file is present whether mteb was installed via ``pip install`` or
# checked out as source. Read once at import — the bytes are ~29 KB and
# don't change at runtime.
_FAVICON_BYTES = (files("mteb.api") / "static" / "favicon.png").read_bytes()


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve the MTEB logo as the browser-tab favicon for every API page.

    Browsers request ``/favicon.ico`` for any URL they navigate to (Swagger,
    ReDoc, the JSON responses) so a single ``.ico`` route is enough. The
    body is actually a PNG — every modern browser accepts a PNG payload
    under the ``.ico`` URL.
    """
    return Response(
        content=_FAVICON_BYTES,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/icon/{name:path}")
async def benchmark_icon(name: str) -> Response:
    """Proxy and long-cache the benchmark's icon.

    Why: upstream icons live on github.com which forces a redirect to
    raw.githubusercontent.com (uncached) and serves the SVG with
    ``max-age=300``. Proxying lets us hand the browser one year of
    ``immutable`` caching, so each user's browser refetches the same flag
    SVGs at most once. ETag + 304 revalidation is provided by the
    app-level ETagMiddleware; this route only sets ``Cache-Control``.
    """
    import mteb

    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not bench.icon:
        raise HTTPException(status_code=404, detail=f"{name} has no icon")

    # Only URL-backed icons go through the proxy. Text/emoji icons are passed
    # back to the client verbatim in the BenchmarkSchema.icon field and never
    # hit this route.
    from mteb.api.schemas import _is_url

    if not _is_url(bench.icon):
        raise HTTPException(status_code=404, detail=f"{name} icon is not a URL")

    cached = await get_icon(name, bench.icon)
    if cached is None:
        raise HTTPException(status_code=502, detail="Upstream icon fetch failed")

    BENCHMARK_SELECTIONS.labels(name=name, endpoint="icon").inc()
    return Response(
        content=cached.body,
        media_type=cached.content_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/benchmarks/menu")
async def benchmarks_menu() -> list[MenuEntrySchema]:
    """Return the nested benchmark menu (groups + benchmarks) used by the frontend nav."""
    return _menu_schemas()


@router.get("/benchmarks")
async def list_benchmarks(include_hidden: bool = False) -> list[BenchmarkSchema]:
    """List benchmarks.

    By default returns only the curated leaderboard set (the same one the
    Gradio app shows). Pass ``?include_hidden=true`` to also include off-menu
    benchmarks (``display_on_leaderboard=False``) so an all-benchmarks page can
    surface them.
    """
    return _benchmark_schemas(include_hidden)


@router.get("/benchmarks/{name:path}/scores")
async def benchmark_scores(name: str) -> BenchmarkSummarySchema:
    """Return the full summary payload (one row per model) for benchmark ``name``.

    Path is ``/scores`` (mirrors ``/tasks/{name}/scores`` and
    ``/models/{name}/scores``); the legacy ``/summary`` URL is kept as
    an alias below for back-compat during the frontend deploy window.
    """
    import mteb

    try:
        mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return await get_summary(name)


@router.get("/benchmarks/{name:path}/summary", include_in_schema=False)
async def benchmark_summary(name: str) -> BenchmarkSummarySchema:
    """Deprecated alias for ``/benchmarks/{name}/scores``.

    Remove once every deployed frontend bundle is on the new path.
    """
    return await benchmark_scores(name)


@router.get("/benchmarks/{name:path}")
async def benchmark_detail(name: str) -> BenchmarkSchema:
    """Return the benchmark's static metadata (tasks, languages, domains, etc.)."""
    import mteb

    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return benchmark_to_schema(bench)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _filtered_task_schemas(
    languages: tuple[str, ...] | None,
    types: tuple[str, ...] | None,
    domains: tuple[str, ...] | None,
    modalities: tuple[str, ...] | None,
    categories: tuple[str, ...] | None,
    name_query: str | None,
) -> list[TaskMetaSchema]:
    # ``mteb.get_tasks(...)`` instantiates every matching task class and runs
    # ``filter_languages`` per task (~2.5s for the full registry). Schemas only
    # need ``task.metadata`` (class-level), so filter the class refs directly
    # and pass them to ``task_to_meta_schema`` — skip instantiation entirely.
    from mteb.filter_tasks import filter_tasks
    from mteb.get_tasks import TASK_LIST

    task_classes = filter_tasks(  # type: ignore[misc]
        TASK_LIST,  # type: ignore[arg-type]
        languages=list(languages) if languages else None,
        task_types=list(types) if types else None,
        domains=list(domains) if domains else None,  # type: ignore[arg-type]
        modalities=list(modalities) if modalities else None,  # type: ignore[arg-type]
        categories=list(categories) if categories else None,  # type: ignore[arg-type]
        exclude_aggregate=True,
    )
    if name_query:
        q = name_query.lower()
        task_classes = [c for c in task_classes if q in c.metadata.name.lower()]
    return [_with_task_num_models(task_to_meta_schema(c)) for c in task_classes]


def _as_tuple(values: list[str] | None) -> tuple[str, ...] | None:
    """Flatten + comma-split + strip a query-string list parameter into a hashable tuple."""
    if not values:
        return None
    flat: list[str] = []
    for v in values:
        for piece in v.split(","):
            stripped = piece.strip()
            if stripped:
                flat.append(stripped)
    return tuple(flat) if flat else None


@router.get("/tasks")
async def list_tasks(
    languages: Annotated[list[str] | None, Query()] = None,
    types: Annotated[list[str] | None, Query()] = None,
    domains: Annotated[list[str] | None, Query()] = None,
    modalities: Annotated[list[str] | None, Query()] = None,
    categories: Annotated[list[str] | None, Query()] = None,
    name: Annotated[str | None, Query()] = None,
) -> list[TaskMetaSchema]:
    """List task metadata, filtered by language / type / domain / modality / category / name substring."""
    return _filtered_task_schemas(
        _as_tuple(languages),
        _as_tuple(types),
        _as_tuple(domains),
        _as_tuple(modalities),
        _as_tuple(categories),
        name,
    )


@router.get("/tasks/{name:path}/scores")
async def task_scores(name: str) -> TaskScoresSchema:
    """Return per-model scores on task ``name`` across every benchmark that hosts it."""
    from mteb.get_tasks import _TASKS_REGISTRY

    if name not in _TASKS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task: {name}")
    TASK_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return await get_task_scores(name)


@router.get("/tasks/{name:path}")
async def task_detail(name: str) -> TaskMetaSchema:
    """Return static metadata for task ``name`` (languages, domains, modalities, description)."""
    from mteb.get_tasks import _TASKS_REGISTRY

    if name not in _TASKS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task: {name}")
    TASK_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return _with_task_num_models(task_to_meta_schema(_TASKS_REGISTRY[name]))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _filtered_model_schemas(
    model_types: tuple[str, ...] | None,
    frameworks: tuple[str, ...] | None,
    open_weights: bool | None,
    instruction_tuned: bool | None,
    min_params_b: float | None,
    max_params_b: float | None,
    modalities: tuple[str, ...] | None,
    exclusive_modality: bool,
    name_query: str | None,
) -> list[ModelMetaSchema]:
    from mteb.models.get_model_meta import get_model_metas

    lower = int(min_params_b * 1e9) if min_params_b is not None else None
    upper = int(max_params_b * 1e9) if max_params_b is not None else None

    metas = get_model_metas(
        open_weights=open_weights,
        frameworks=list(frameworks) if frameworks else None,
        n_parameters_range=(lower, upper),
        use_instructions=instruction_tuned,
        model_types=list(model_types) if model_types else None,
        modalities=list(modalities) if modalities else None,  # type: ignore[arg-type]
        exclusive_modality_filter=exclusive_modality,
    )
    if name_query:
        q = name_query.lower()
        metas = [m for m in metas if m.name and q in m.name.lower()]
    return [model_meta_to_schema(m, zero_shot_pct=None) for m in metas]


@router.get("/models")
async def list_models(
    model_types: Annotated[list[str] | None, Query()] = None,
    frameworks: Annotated[list[str] | None, Query()] = None,
    open_weights: Annotated[bool | None, Query()] = None,
    instruction_tuned: Annotated[bool | None, Query()] = None,
    min_params_b: Annotated[float | None, Query()] = None,
    max_params_b: Annotated[float | None, Query()] = None,
    modalities: Annotated[list[str] | None, Query()] = None,
    exclusive_modality: Annotated[bool, Query()] = False,
    name: Annotated[str | None, Query()] = None,
) -> list[ModelMetaSchema]:
    """List model metadata, filtered by type / framework / size / modality / name substring."""
    return _filtered_model_schemas(
        _as_tuple(model_types),
        _as_tuple(frameworks),
        open_weights,
        instruction_tuned,
        min_params_b,
        max_params_b,
        _as_tuple(modalities),
        exclusive_modality,
        name,
    )


@router.get("/models/{name:path}/scores")
async def model_scores(name: str) -> ModelScoresSchema:
    """Return per-benchmark scores for model ``name`` across every leaderboard benchmark."""
    from mteb.models.model_implementations import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    try:
        result = await get_model_scores(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    MODEL_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return result


@router.get("/models/{name:path}")
async def model_detail(name: str) -> ModelMetaSchema:
    """Return static metadata for model ``name`` (params, embed dim, framework, etc.)."""
    from mteb.models.get_model_meta import get_model_meta
    from mteb.models.model_implementations import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    MODEL_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return model_meta_to_schema(get_model_meta(name))
