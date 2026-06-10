"""FastAPI routes for the leaderboard.

Cached endpoints serve pre-built JSON bytes via :func:`_cached_json`, which
also handles 304 revalidation, gzip negotiation, and ``Cache-Control``.
Non-cached endpoints return pydantic schemas directly — FastAPI serialises
them via pydantic-core.
"""

from __future__ import annotations

import functools
import json
import logging
from collections import OrderedDict
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Any

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
from mteb.api.aggregators import build_benchmark_leaders
from mteb.api.cache import (
    _cached_bytes,
    get_model_scores_bytes,
    get_per_language_bytes,
    get_summary_bytes,
    get_task_scores_bytes,
)
from mteb.api.frames import _load_per_benchmark_frames
from mteb.api.icons import get_icon
from mteb.api.metrics import (
    BENCHMARK_SELECTIONS,
    MODEL_SELECTIONS,
    TASK_SELECTIONS,
    render_metrics,
)
from mteb.api.schemas import (
    BenchmarkLeadersSchema,
    BenchmarkPerLanguageSchema,
    BenchmarkSchema,
    BenchmarkSummarySchema,
    MenuEntrySchema,
    ModelMetaSchema,
    ModelScoresSchema,
    TaskMetaSchema,
    TaskScoresSchema,
    _is_url,
)
from mteb.api.serialization import (
    serialize_bytes,
)
from mteb.benchmarks._leaderboard_menu import HOME_BENCHMARK_ENTRIES
from mteb.filter_tasks import filter_tasks
from mteb.get_tasks import _TASKS_REGISTRY, TASK_LIST
from mteb.models.get_model_meta import get_model_meta, get_model_metas
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    import asyncio

    from mteb.api.serialization import (
        Serialized,
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
    """Return a JSON response from a cached `Serialized` payload.

    Short-circuits to ``304 Not Modified`` when the client revalidates with a
    matching ``If-None-Match``. Picks the pre-gzipped body when the client
    advertises ``Accept-Encoding: gzip`` so GZipMiddleware can skip
    compressing. ``max_age`` defaults to 4 hours — pass ``None`` for
    ``Cache-Control: no-cache``.
    """
    cache_control = f"public, max-age={max_age}" if max_age is not None else "no-cache"
    headers = {
        "etag": payload.etag,
        "vary": "accept-encoding",
        "cache-control": cache_control,
    }
    if request.headers.get("if-none-match") == payload.etag:
        return Response(status_code=304, headers=headers)
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


@functools.cache
def _benchmark_name_set() -> frozenset[str]:
    """All registered benchmark names — for fast 404 validation."""
    return frozenset(b.name for b in mteb.get_benchmarks())


def _require_benchmark(name: str) -> None:
    if name not in _benchmark_name_set():
        raise HTTPException(status_code=404, detail=f"Unknown benchmark: {name}")


def _require_task(name: str) -> None:
    if name not in _TASKS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task: {name}")


def _require_model(name: str) -> None:
    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")


# Narrow exception set: a parquet-load failure should fall back to empty maps,
# but programmer bugs (TypeError, AttributeError) must surface.
_FRAME_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    pl.exceptions.PolarsError,
)


def _safe_load_frames(
    label: str,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame] | None:
    """Return the per-benchmark + unified frames, or ``None`` on a load failure.

    Warns with ``label`` so each caller's log line distinguishes which map
    couldn't be built. Programmer bugs (TypeError, AttributeError) propagate
    so they aren't silently absorbed.
    """
    try:
        return _load_per_benchmark_frames()
    except _FRAME_LOAD_ERRORS as exc:
        logger.warning("%s unavailable: %s: %s", label, type(exc).__name__, exc)
        return None


@functools.cache
def _num_models_map() -> dict[str, int]:
    """{benchmark_name -> distinct model count}, cached for the process lifetime."""
    loaded = _safe_load_frames("num_models map")
    if loaded is None:
        return {}
    frames, _ = loaded
    return {
        name: int(frame["model_name"].n_unique())
        for name, frame in frames.items()
        if "model_name" in frame.columns
    }


@functools.cache
def _task_num_models_map() -> dict[str, int]:
    """{task_name -> distinct model count}, cached for the process lifetime."""
    loaded = _safe_load_frames("task num_models map")
    if loaded is None:
        return {}
    _, unified = loaded
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
            patched_child: BenchmarkSchema | MenuEntrySchema
            if isinstance(child, BenchmarkSchema):
                patched_child = _with_num_models(child)
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


@functools.cache
def _menu_schemas() -> list[MenuEntrySchema]:
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


def _serialize_schemas(schemas: list[Any], adapter: TypeAdapter[Any]) -> Serialized:
    """Dump a schema list to bytes (+gzip+ETag) via the given TypeAdapter."""
    return serialize_bytes(adapter.dump_json(schemas, by_alias=True))


@functools.cache
def _menu_schemas_bytes() -> Serialized:
    return _serialize_schemas(_menu_schemas(), _MENU_LIST_ADAPTER)


@functools.lru_cache(maxsize=2)
def _benchmark_schemas_bytes(include_hidden: bool = False) -> Serialized:
    return _serialize_schemas(
        _benchmark_schemas(include_hidden), _BENCHMARK_LIST_ADAPTER
    )


@functools.lru_cache(maxsize=128)
def _filtered_task_schemas_bytes(
    languages: tuple[str, ...] | None,
    types: tuple[str, ...] | None,
    domains: tuple[str, ...] | None,
    modalities: tuple[str, ...] | None,
    categories: tuple[str, ...] | None,
) -> Serialized:
    return _serialize_schemas(
        _filtered_task_schemas(languages, types, domains, modalities, categories),
        _TASK_LIST_ADAPTER,
    )


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
    return _serialize_schemas(
        _filtered_model_schemas(
            model_types,
            frameworks,
            open_weights,
            instruction_tuned,
            min_params_b,
            max_params_b,
            modalities,
            exclusive_modality,
        ),
        _MODEL_LIST_ADAPTER,
    )


@infra_router.get("/health")
async def health() -> dict[str, bool]:
    """Liveness probe."""
    return {"ok": True}


@infra_router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@infra_router.get("/robots.txt", include_in_schema=False)
async def robots_txt() -> Response:
    """Cached ``Allow: /`` body so crawler probes stop 404-ing."""
    return Response(
        content="User-agent: *\nAllow: /\n",
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


@router.get("/icon/{name:path}", include_in_schema=False)
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


# LRU-bounded: ?buckets= can take arbitrary tuples, so callers that vary the
# query would otherwise grow the bytes + locks dicts without limit.
_LEADER_BYTES_MAX = 256
_leader_bytes: OrderedDict[
    tuple[str, tuple[tuple[float, float | None], ...]], Serialized
] = OrderedDict()
_leader_bytes_locks: dict[
    tuple[str, tuple[tuple[float, float | None], ...]], asyncio.Lock
] = {}


@router.get("/benchmarks/{name:path}/leaders", response_model=BenchmarkLeadersSchema)
async def benchmark_leaders(
    request: Request,
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
) -> Response:
    """Highest-mean-task model in each size bucket — slim payload for home tiles."""
    _require_benchmark(name)
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="leaders").inc()

    parsed = _parse_buckets_json(buckets)
    key = (name, tuple(parsed))

    async def _build() -> BenchmarkLeadersSchema:
        return await build_benchmark_leaders(name, parsed)

    payload = await _cached_bytes(
        _leader_bytes,
        _leader_bytes_locks,
        key,
        _build,
        layer="leaders",
        max_size=_LEADER_BYTES_MAX,
    )
    return _cached_json(request, payload)


@router.get("/benchmarks/{name:path}", response_model=BenchmarkSchema)
async def benchmark_detail(name: str) -> BenchmarkSchema:
    """Static metadata for benchmark ``name``."""
    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    BENCHMARK_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return _with_num_models(benchmark_to_schema(bench))


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
    # Name filter is high-cardinality but typeahead repeats heavily —
    # cache by (filter_key, name_lower) so each keystroke is a hit not a rebuild.
    return _cached_json(
        request, _filtered_task_schemas_bytes_named(filter_key, name.lower())
    )


@functools.lru_cache(maxsize=256)
def _filtered_task_schemas_bytes_named(
    filter_key: tuple[
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
    ],
    name_lower: str,
) -> Serialized:
    return _serialize_schemas(
        [
            s
            for s in _filtered_task_schemas(*filter_key)
            if name_lower in s.name.lower()
        ],
        _TASK_LIST_ADAPTER,
    )


@router.get("/tasks/{name:path}/scores", response_model=TaskScoresSchema)
async def task_scores(request: Request, name: str) -> Response:
    """Per-model scores on task ``name`` across every benchmark that hosts it."""
    _require_task(name)
    TASK_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return _cached_json(request, await get_task_scores_bytes(name))


@router.get("/tasks/{name:path}", response_model=TaskMetaSchema)
async def task_detail(name: str) -> TaskMetaSchema:
    """Static metadata for task ``name``."""
    _require_task(name)
    TASK_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return _with_task_num_models(task_to_meta_schema(_TASKS_REGISTRY[name]))


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
    return _cached_json(
        request, _filtered_model_schemas_bytes_named(filter_key, name.lower())
    )


@functools.lru_cache(maxsize=256)
def _filtered_model_schemas_bytes_named(
    filter_key: tuple[
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        bool | None,
        bool | None,
        float | None,
        float | None,
        tuple[str, ...] | None,
        bool,
    ],
    name_lower: str,
) -> Serialized:
    return _serialize_schemas(
        [
            s
            for s in _filtered_model_schemas(*filter_key)
            if name_lower in s.name.lower()
        ],
        _MODEL_LIST_ADAPTER,
    )


@router.get("/models/{name:path}/scores", response_model=ModelScoresSchema)
async def model_scores(request: Request, name: str) -> Response:
    """Per-benchmark scores for model ``name``."""
    _require_model(name)
    MODEL_SELECTIONS.labels(name=name, endpoint="scores").inc()
    return _cached_json(request, await get_model_scores_bytes(name))


@router.get("/models/{name:path}", response_model=ModelMetaSchema)
async def model_detail(name: str) -> ModelMetaSchema:
    """Static metadata for model ``name``."""
    _require_model(name)
    MODEL_SELECTIONS.labels(name=name, endpoint="detail").inc()
    return model_meta_to_schema(get_model_meta(name))
