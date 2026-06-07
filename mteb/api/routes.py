"""FastAPI routes for the leaderboard.

Cached endpoints serve pre-built JSON bytes via :func:`_cached_json`, which
also handles 304 revalidation, gzip negotiation, and ``Cache-Control``.
Non-cached endpoints return pydantic schemas directly — FastAPI serialises
them via pydantic-core.
"""

from __future__ import annotations

import functools
import logging
from importlib.resources import files
from typing import Annotated

import polars as pl
from fastapi import (  # noqa: TC002 — Request needed at runtime for FastAPI DI
    APIRouter,
    HTTPException,
    Query,
    Request,
    Response,
)
from pydantic import TypeAdapter

import mteb
from mteb.api.adapters import (
    benchmark_to_schema,
    menus_to_schemas,
    model_meta_to_schema,
    task_to_meta_schema,
)
from mteb.api.cache import (  # noqa: TC001 — Serialized is a runtime type annotation
    Serialized,
    get_model_scores_bytes,
    get_per_language_bytes,
    get_summary_bytes,
    get_task_scores_bytes,
    serialize_bytes,
)
from mteb.api.icons import get_icon
from mteb.api.metrics import (
    BENCHMARK_SELECTIONS,
    MODEL_SELECTIONS,
    TASK_SELECTIONS,
    render_metrics,
)
from mteb.api.schemas import (  # noqa: TC001 — FastAPI inspects return annotations at registration
    BenchmarkLeadersSchema,
    BenchmarkPerLanguageSchema,
    BenchmarkSchema,
    BenchmarkSummarySchema,
    MenuEntrySchema,
    ModelMetaSchema,
    ModelScoresSchema,
    TaskMetaSchema,
    TaskScoresSchema,
)

logger = logging.getLogger(__name__)
# Root-level infra (health / metrics / asset proxies); not under /v1.
infra_router = APIRouter()
router = APIRouter()


# Data changes only when the process reloads the parquet (server restart); a
# 4-hour browser cache lets repeat hits skip the network entirely. ETag still
# drives 304 revalidation after max-age expires, so a deploy that ships fresh
# data isn't blocked by a stale cache for long.
_DEFAULT_MAX_AGE = 4 * 60 * 60


def _cached_json(
    request: Request, payload: Serialized, *, max_age: int | None = _DEFAULT_MAX_AGE
) -> Response:
    """Return a JSON response from a cached :class:`Serialized` payload.

    Short-circuits to ``304 Not Modified`` when the client revalidates with a
    matching ``If-None-Match``. Picks the pre-gzipped body when the client
    advertises ``Accept-Encoding: gzip`` so GZipMiddleware can skip
    compressing. ``max_age`` defaults to 4 hours — pass ``None`` for
    ``Cache-Control: no-cache``.
    """
    cache_control = f"public, max-age={max_age}" if max_age is not None else "no-cache"
    if request.headers.get("if-none-match") == payload.etag:
        return Response(
            status_code=304,
            headers={"etag": payload.etag, "cache-control": cache_control},
        )
    headers = {
        "etag": payload.etag,
        "vary": "accept-encoding",
        "cache-control": cache_control,
    }
    use_gzip = (
        payload.body_gzip is not None
        and "gzip" in request.headers.get("accept-encoding", "").lower()
    )
    if use_gzip:
        headers["content-encoding"] = "gzip"
        body = payload.body_gzip
    else:
        body = payload.body
    return Response(content=body, media_type="application/json", headers=headers)


@functools.lru_cache(maxsize=1)
def _benchmark_name_set() -> frozenset[str]:
    """All registered benchmark names — for fast 404 validation."""
    return frozenset(b.name for b in mteb.get_benchmarks())


def _require_benchmark(name: str) -> None:
    if name not in _benchmark_name_set():
        raise HTTPException(status_code=404, detail=f"Unknown benchmark: {name}")


@functools.lru_cache(maxsize=1)
def _num_models_map() -> dict[str, int]:
    """{benchmark_name -> distinct model count}, cached for the process lifetime."""
    from mteb.api.cache import _load_per_benchmark_frames

    try:
        frames, _ = _load_per_benchmark_frames()
    except Exception:
        return {}
    return {
        name: int(frame["model_name"].n_unique())
        for name, frame in frames.items()
        if "model_name" in frame.columns
    }


@functools.lru_cache(maxsize=1)
def _task_num_models_map() -> dict[str, int]:
    """{task_name -> distinct model count}, cached for the process lifetime."""
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
    return dict(zip(grouped["task_name"].to_list(), (int(n) for n in grouped["n"])))


def _with_num_models(schema: BenchmarkSchema) -> BenchmarkSchema:
    n = _num_models_map().get(schema.name, 0)
    if n == schema.num_models:
        return schema
    return schema.model_copy(update={"num_models": n})


def _with_task_num_models(schema: TaskMetaSchema) -> TaskMetaSchema:
    n = _task_num_models_map().get(schema.name, 0)
    if n == schema.num_models:
        return schema
    return schema.model_copy(update={"num_models": n})


def _patch_menu_counts(entries: list[MenuEntrySchema]) -> list[MenuEntrySchema]:
    """Overlay ``num_models`` onto every benchmark child in the menu tree.

    Reuses the input entry when no descendant changed — avoids needless
    pydantic ``model_copy`` allocations during prewarm.
    """
    patched: list[MenuEntrySchema] = []
    for entry in entries:
        new_children: list[BenchmarkSchema | MenuEntrySchema] = []
        any_changed = False
        for child in entry.children:
            if isinstance(child, BenchmarkSchema):
                patched_child = _with_num_models(child)
                if patched_child is not child:
                    any_changed = True
                new_children.append(patched_child)
            else:
                [patched_child] = _patch_menu_counts([child])
                if patched_child is not child:
                    any_changed = True
                new_children.append(patched_child)
        if any_changed:
            patched.append(entry.model_copy(update={"children": new_children}))
        else:
            patched.append(entry)
    return patched


@functools.lru_cache(maxsize=1)
def _menu_schemas() -> list[MenuEntrySchema]:
    from mteb.benchmarks._leaderboard_menu import HOME_BENCHMARK_ENTRIES

    raw = menus_to_schemas(list(HOME_BENCHMARK_ENTRIES))
    return _patch_menu_counts(raw)


@functools.lru_cache(maxsize=2)
def _benchmark_schemas(include_hidden: bool = False) -> list[BenchmarkSchema]:
    benches = (
        mteb.get_benchmarks()
        if include_hidden
        else mteb.get_benchmarks(display_on_leaderboard=True)
    )
    return [_with_num_models(benchmark_to_schema(b)) for b in benches]


# TypeAdapters dump a list of pydantic models to JSON bytes in pydantic-core
# (Rust) without the Python-level iteration FastAPI's response encoder does.
_MENU_LIST_ADAPTER = TypeAdapter(list[MenuEntrySchema])
_BENCHMARK_LIST_ADAPTER = TypeAdapter(list[BenchmarkSchema])
_TASK_LIST_ADAPTER = TypeAdapter(list[TaskMetaSchema])
_MODEL_LIST_ADAPTER = TypeAdapter(list[ModelMetaSchema])


@functools.lru_cache(maxsize=1)
def _menu_schemas_bytes() -> Serialized:
    return serialize_bytes(_MENU_LIST_ADAPTER.dump_json(_menu_schemas(), by_alias=True))


@functools.lru_cache(maxsize=2)
def _benchmark_schemas_bytes(include_hidden: bool = False) -> Serialized:
    return serialize_bytes(
        _BENCHMARK_LIST_ADAPTER.dump_json(
            _benchmark_schemas(include_hidden), by_alias=True
        )
    )


@functools.lru_cache(maxsize=128)
def _filtered_task_schemas_bytes(
    languages: tuple[str, ...] | None,
    types: tuple[str, ...] | None,
    domains: tuple[str, ...] | None,
    modalities: tuple[str, ...] | None,
    categories: tuple[str, ...] | None,
) -> Serialized:
    schemas = _filtered_task_schemas(languages, types, domains, modalities, categories)
    return serialize_bytes(_TASK_LIST_ADAPTER.dump_json(schemas, by_alias=True))


@functools.lru_cache(maxsize=128)
def _filtered_model_schemas_bytes(
    model_types: tuple[str, ...] | None,
    frameworks: tuple[str, ...] | None,
    open_weights: bool | None,
    instruction_tuned: bool | None,
    min_params_b: float | None,
    max_params_b: float | None,
    modalities: tuple[str, ...] | None,
    exclusive_modality: bool,
) -> Serialized:
    schemas = _filtered_model_schemas(
        model_types,
        frameworks,
        open_weights,
        instruction_tuned,
        min_params_b,
        max_params_b,
        modalities,
        exclusive_modality,
    )
    return serialize_bytes(_MODEL_LIST_ADAPTER.dump_json(schemas, by_alias=True))


@infra_router.get("/health")
async def health() -> dict[str, bool]:
    """Liveness probe."""
    return {"ok": True}


@infra_router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


_ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


@infra_router.get("/robots.txt", include_in_schema=False)
async def robots_txt() -> Response:
    """Cached ``Disallow: /`` body so crawler probes stop 404-ing."""
    return Response(
        content=_ROBOTS_TXT,
        media_type="text/plain",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# Shipped as package data via pyproject's ``[tool.setuptools.package-data]``.
_FAVICON_BYTES = (files("mteb.api") / "static" / "favicon.png").read_bytes()


@infra_router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve the MTEB logo (PNG payload under the .ico URL)."""
    return Response(
        content=_FAVICON_BYTES,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/icon/{name:path}")
async def benchmark_icon(name: str) -> Response:
    """Proxy and long-cache the benchmark's icon.

    Upstream icons live on github.com which redirects to raw.githubusercontent
    (uncached, ``max-age=300``). Proxying lets us hand the browser one year of
    ``immutable`` caching.
    """
    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not bench.icon:
        raise HTTPException(status_code=404, detail=f"{name} has no icon")

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


@router.get("/benchmarks/menu", response_model=list[MenuEntrySchema])
async def benchmarks_menu(request: Request) -> Response:
    """Nested benchmark menu used by the frontend nav."""
    return _cached_json(request, _menu_schemas_bytes())


@router.get("/benchmarks", response_model=list[BenchmarkSchema])
async def list_benchmarks(request: Request, include_hidden: bool = False) -> Response:
    """List benchmarks; ``?include_hidden=true`` adds off-menu entries."""
    return _cached_json(request, _benchmark_schemas_bytes(include_hidden))


@router.get("/benchmarks/{name:path}/scores", response_model=BenchmarkSummarySchema)
async def benchmark_scores(
    request: Request,
    name: str,
    languages: Annotated[
        list[str] | None,
        Query(
            description=(
                "Restrict the summary aggregation to subsets whose language list "
                "intersects this set. Comma-separated or repeated; accepts either "
                "raw codes (`eng-Latn`) or human labels (`English`)."
            ),
        ),
    ] = None,
) -> Response:
    """Full summary payload (one row per model) for benchmark ``name``.

    When ``languages=`` is set, the long results frame is pre-filtered to
    subsets covering those languages before the summary builders run.
    """
    _require_benchmark(name)
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="scores").inc()
    # Sort+dedupe so the cache key is order-independent.
    picked = tuple(sorted(set(_as_tuple(languages) or ())))
    return _cached_json(request, await get_summary_bytes(name, picked))


@router.get(
    "/benchmarks/{name:path}/summary",
    include_in_schema=False,
    response_model=BenchmarkSummarySchema,
)
async def benchmark_summary(request: Request, name: str) -> Response:
    """Deprecated alias for ``/benchmarks/{name}/scores`` — kept during frontend rollout."""
    return await benchmark_scores(request, name)


@router.get(
    "/benchmarks/{name:path}/per-language",
    response_model=BenchmarkPerLanguageSchema,
)
async def benchmark_per_language(request: Request, name: str) -> Response:
    """Per-(model, language) mean main_score, lazy-loaded by the Per-language tab."""
    _require_benchmark(name)
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="per-language").inc()
    return _cached_json(request, await get_per_language_bytes(name))


def _parse_buckets_json(raw: str) -> list[tuple[float, float | None]]:
    """Parse the ``buckets`` query param.

    Wire format: JSON array of ``[min, max]`` pairs in millions of parameters.
    ``null`` (or a 1-element entry) means open-ended top bucket. Example:
    ``[[0,500],[500,1000],[1000,5000],[5000,null]]``.

    Returned tuples stay in MILLIONS — callers convert when comparing against
    ``ModelMeta.total_params_b`` (billions).
    """
    import json

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid `buckets` JSON: {exc.msg}"
        ) from exc
    if not isinstance(decoded, list) or not decoded:
        raise HTTPException(
            status_code=422,
            detail="`buckets` must be a non-empty JSON array of [min, max] pairs",
        )
    if len(decoded) > 8:
        raise HTTPException(status_code=422, detail="`buckets` is capped at 8 entries")

    out: list[tuple[float, float | None]] = []
    for i, entry in enumerate(decoded):
        if not isinstance(entry, list) or not (1 <= len(entry) <= 2):
            raise HTTPException(
                status_code=422,
                detail=f"`buckets[{i}]` must be a 1- or 2-element array (got {entry!r})",
            )
        lo_raw = entry[0]
        if not isinstance(lo_raw, (int, float)):
            raise HTTPException(
                status_code=422, detail=f"`buckets[{i}][0]` must be a number"
            )
        lo = float(lo_raw)
        if lo < 0:
            raise HTTPException(
                status_code=422, detail=f"`buckets[{i}][0]` must be ≥ 0"
            )
        hi: float | None = None
        if len(entry) == 2 and entry[1] is not None:
            hi_raw = entry[1]
            if not isinstance(hi_raw, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"`buckets[{i}][1]` must be a number or null",
                )
            hi = float(hi_raw)
            if hi <= lo:
                raise HTTPException(
                    status_code=422,
                    detail=f"`buckets[{i}]`: max ({hi}) must be > min ({lo})",
                )
        out.append((lo, hi))
    return out


@router.get("/benchmarks/{name:path}/leaders")
async def benchmark_leaders(
    name: str,
    buckets: Annotated[
        str,
        Query(
            description=(
                "JSON array of [min, max] tuples in millions of parameters. "
                "Use null (or omit) for an open-ended top bucket. "
                "Example: `?buckets=[[0,500],[500,1000],[1000,5000],[5000,null]]`."
            ),
        ),
    ],
) -> BenchmarkLeadersSchema:
    """Highest-mean-task model in each size bucket — slim payload for home tiles."""
    _require_benchmark(name)
    from mteb.api.aggregators import build_benchmark_leaders

    parsed = _parse_buckets_json(buckets)
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="leaders").inc()
    return await build_benchmark_leaders(name, parsed)


@router.get("/benchmarks/{name:path}")
async def benchmark_detail(name: str) -> BenchmarkSchema:
    """Static metadata for benchmark ``name``."""
    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return _with_num_models(benchmark_to_schema(bench))


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
) -> list[TaskMetaSchema]:
    # Filter class refs directly — get_tasks() instantiates and runs
    # filter_languages per task (~2.5s) for metadata we don't need here.
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
    return [_with_task_num_models(task_to_meta_schema(c)) for c in task_classes]


def _as_tuple(values: list[str] | None) -> tuple[str, ...] | None:
    """Flatten + comma-split + strip a list query param into a hashable tuple."""
    if not values:
        return None
    flat: list[str] = []
    for v in values:
        for piece in v.split(","):
            stripped = piece.strip()
            if stripped:
                flat.append(stripped)
    return tuple(flat) if flat else None


@router.get("/tasks", response_model=list[TaskMetaSchema])
async def list_tasks(
    request: Request,
    languages: Annotated[list[str] | None, Query()] = None,
    types: Annotated[list[str] | None, Query()] = None,
    domains: Annotated[list[str] | None, Query()] = None,
    modalities: Annotated[list[str] | None, Query()] = None,
    categories: Annotated[list[str] | None, Query()] = None,
    name: Annotated[str | None, Query()] = None,
) -> Response:
    """List task metadata, filtered by language / type / domain / modality / category / name."""
    filter_key = (
        _as_tuple(languages),
        _as_tuple(types),
        _as_tuple(domains),
        _as_tuple(modalities),
        _as_tuple(categories),
    )
    if not name:
        return _cached_json(request, _filtered_task_schemas_bytes(*filter_key))
    # Name filter is high-cardinality; serialize the filtered subset on the fly.
    q = name.lower()
    schemas = [s for s in _filtered_task_schemas(*filter_key) if q in s.name.lower()]
    return _cached_json(
        request,
        serialize_bytes(_TASK_LIST_ADAPTER.dump_json(schemas, by_alias=True)),
    )


@router.get("/tasks/{name:path}/scores", response_model=TaskScoresSchema)
async def task_scores(request: Request, name: str) -> Response:
    """Per-model scores on task ``name`` across every benchmark that hosts it."""
    from mteb.get_tasks import _TASKS_REGISTRY

    if name not in _TASKS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task: {name}")
    TASK_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return _cached_json(request, await get_task_scores_bytes(name))


@router.get("/tasks/{name:path}")
async def task_detail(name: str) -> TaskMetaSchema:
    """Static metadata for task ``name``."""
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
    return [model_meta_to_schema(m) for m in metas]


@router.get("/models", response_model=list[ModelMetaSchema])
async def list_models(  # noqa: PLR0913, PLR0917 — FastAPI Query params can't be grouped
    request: Request,
    model_types: Annotated[list[str] | None, Query()] = None,
    frameworks: Annotated[list[str] | None, Query()] = None,
    open_weights: Annotated[bool | None, Query()] = None,
    instruction_tuned: Annotated[bool | None, Query()] = None,
    min_params_b: Annotated[float | None, Query()] = None,
    max_params_b: Annotated[float | None, Query()] = None,
    modalities: Annotated[list[str] | None, Query()] = None,
    exclusive_modality: Annotated[bool, Query()] = False,
    name: Annotated[str | None, Query()] = None,
) -> Response:
    """List model metadata, filtered by type / framework / size / modality / name."""
    filter_key = (
        _as_tuple(model_types),
        _as_tuple(frameworks),
        open_weights,
        instruction_tuned,
        min_params_b,
        max_params_b,
        _as_tuple(modalities),
        exclusive_modality,
    )
    if not name:
        return _cached_json(request, _filtered_model_schemas_bytes(*filter_key))
    q = name.lower()
    schemas = [s for s in _filtered_model_schemas(*filter_key) if q in s.name.lower()]
    return _cached_json(
        request,
        serialize_bytes(_MODEL_LIST_ADAPTER.dump_json(schemas, by_alias=True)),
    )


@router.get("/models/{name:path}/scores", response_model=ModelScoresSchema)
async def model_scores(request: Request, name: str) -> Response:
    """Per-benchmark scores for model ``name``."""
    from mteb.models.model_implementations import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    try:
        payload = await get_model_scores_bytes(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    MODEL_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return _cached_json(request, payload)


@router.get("/models/{name:path}")
async def model_detail(name: str) -> ModelMetaSchema:
    """Static metadata for model ``name``."""
    from mteb.models.get_model_meta import get_model_meta
    from mteb.models.model_implementations import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    MODEL_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return model_meta_to_schema(get_model_meta(name))
