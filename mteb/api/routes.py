"""FastAPI routes for the leaderboard.

Cached endpoints serve pre-built JSON bytes via :func:`_cached_json`, which
handles 304 revalidation, gzip negotiation, and ``Cache-Control``.
"""

from __future__ import annotations

import email.utils
import functools
import json
import logging
import time
from collections import OrderedDict
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Any

import polars as pl
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,  # noqa: TC002
    Response,
)
from pydantic import TypeAdapter

import mteb
from mteb.api._errors import FRAME_LOAD_ERRORS
from mteb.api.adapters import (
    benchmark_to_schema,
    menus_to_schemas,
    model_meta_to_schema,
    task_to_meta_schema,
)
from mteb.api.aggregators import build_benchmark_leaders
from mteb.api.cache import (
    CacheLayer,
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
from mteb.api.settings import get_settings
from mteb.benchmarks._leaderboard_menu import HOME_BENCHMARK_ENTRIES
from mteb.filter_tasks import filter_tasks
from mteb.get_tasks import _TASKS_REGISTRY, TASK_LIST
from mteb.models.get_model_meta import get_model_meta, get_model_metas
from mteb.models.model_implementations import MODEL_REGISTRY
from mteb.types.statistics import DescriptiveStatsValue

if TYPE_CHECKING:
    from mteb.api.serialization import (
        Serialized,
    )
logger = logging.getLogger(__name__)
# Root-level infra (health / metrics / asset proxies); not under /v1.
infra_router = APIRouter()
router = APIRouter()

# All cached payloads derive from data loaded at process start. Pre-formatted
# so ``If-Modified-Since`` comparison is byte-equal, not parse-and-compare.
_PROCESS_START_HTTP_DATE: str = email.utils.formatdate(time.time(), usegmt=True)


def _cached_json(request: Request, payload: Serialized) -> Response:
    """JSON response from a cached `Serialized` payload (handles 304 + gzip + Cache-Control)."""
    max_age = get_settings().http_max_age
    cache_control = f"public, max-age={max_age}" if max_age > 0 else "no-cache"
    headers = {
        "etag": payload.etag,
        "last-modified": _PROCESS_START_HTTP_DATE,
        "vary": "accept-encoding",
        "cache-control": cache_control,
    }
    if request.headers.get("if-none-match") == payload.etag or (
        request.headers.get("if-modified-since") == _PROCESS_START_HTTP_DATE
    ):
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


def _safe_load_frames(
    label: str,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame] | None:
    """Per-benchmark + unified frames, or ``None`` on load failure (logged with ``label``)."""
    try:
        return _load_per_benchmark_frames()
    except FRAME_LOAD_ERRORS as exc:
        logger.warning("%s unavailable: %s: %s", label, type(exc).__name__, exc)
        return None


@functools.cache
def _num_models_map() -> dict[str, int]:
    """{benchmark_name -> count of fully-evaluated models}."""
    loaded = _safe_load_frames("num_models map")
    if loaded is None:
        return {}
    frames, _ = loaded
    expected_tasks = {b.name: len(b.tasks) for b in mteb.get_benchmarks()}
    out: dict[str, int] = {}
    for name, frame in frames.items():
        total = expected_tasks.get(name, 0)
        if total <= 0 or "model_name" not in frame.columns:
            continue
        if "task_name" not in frame.columns:
            continue
        fully_evaluated = (
            frame.lazy()
            .drop_nulls("score")
            .group_by("model_name")
            .agg(pl.col("task_name").n_unique().alias("n_tasks"))
            .filter(pl.col("n_tasks") >= total)
            .select(pl.len())
            .collect()
            .item()
        )
        out[name] = int(fully_evaluated)
    return out


@functools.cache
def _task_num_models_map() -> dict[str, int]:
    """{task_name -> count of fully-evaluated models}."""
    loaded = _safe_load_frames("task num_models map")
    if loaded is None:
        return {}
    _, unified = loaded
    if unified.is_empty() or "task_name" not in unified.columns:
        return {}
    triples: list[tuple[str, str, str]] = []
    for name, cls in _TASKS_REGISTRY.items():
        splits = list(cls.metadata.eval_splits) or ["default"]
        subsets = list(cls.metadata.hf_subsets) or ["default"]
        for sp in splits:
            for ss in subsets:
                triples.append((name, sp, ss))
    if not triples:
        return {}
    expected_df = pl.DataFrame(
        triples,
        schema={"task_name": pl.Utf8, "split": pl.Utf8, "subset": pl.Utf8},
        orient="row",
    )
    expected_per_task = (
        expected_df.lazy().group_by("task_name").agg(pl.len().alias("expected_cells"))
    )
    grouped = (
        unified.lazy()
        .drop_nulls("score")
        .join(expected_df.lazy(), on=["task_name", "split", "subset"], how="inner")
        .group_by(["task_name", "model_name"])
        .agg(pl.len().alias("n_cells"))
        .join(expected_per_task, on="task_name", how="inner")
        .filter(pl.col("n_cells") >= pl.col("expected_cells"))
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
    """Overlay ``num_models`` onto benchmark children; reuse entries when unchanged."""
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


# TypeAdapters dump model lists to JSON in pydantic-core (Rust) — faster than
# FastAPI's per-item response encoder.
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


@functools.cache
def _favicon_bytes() -> bytes:
    """Lazy-load favicon so a missing PNG fails at the route, not startup."""
    return (files("mteb.api") / "static" / "favicon.png").read_bytes()


@infra_router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve the MTEB logo (PNG payload under the .ico URL)."""
    return Response(
        content=_favicon_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/icon/{name:path}", include_in_schema=False)
async def benchmark_icon(name: str) -> Response:
    """Proxy upstream icon with one-year ``immutable`` Cache-Control."""
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
    """Full summary payload for benchmark ``name``; ``languages=`` pre-filters subsets."""
    _require_benchmark(name)
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
    return _cached_json(request, await get_per_language_bytes(name))


def _parse_buckets_json(raw: str) -> list[tuple[float, float | None]]:
    """Parse the ``buckets`` query param: JSON ``[[min, max], ...]`` in millions of params.

    ``null`` (or a 1-element entry) means open-ended top bucket.
    Returned tuples stay in MILLIONS — callers convert to billions.
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


_leader_bytes: CacheLayer[
    tuple[str, tuple[tuple[float, float | None], ...]], Serialized
] = CacheLayer(name="leaders", store=OrderedDict(), max_size=256)


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

    parsed = _parse_buckets_json(buckets)
    key = (name, tuple(parsed))

    async def _build() -> BenchmarkLeadersSchema:
        return await build_benchmark_leaders(name, parsed)

    payload = await _cached_bytes(_leader_bytes, key, _build)
    return _cached_json(request, payload)


@router.get("/benchmarks/{name:path}", response_model=BenchmarkSchema)
async def benchmark_detail(name: str) -> BenchmarkSchema:
    """Static metadata for benchmark ``name``."""
    try:
        bench = mteb.get_benchmark(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    BENCHMARK_SELECTIONS.labels(name=name).inc()
    return _with_num_models(benchmark_to_schema(bench))


@functools.lru_cache(maxsize=128)
def _filtered_task_schemas(
    languages: tuple[str, ...] | None,
    types: tuple[str, ...] | None,
    domains: tuple[str, ...] | None,
    modalities: tuple[str, ...] | None,
    categories: tuple[str, ...] | None,
) -> list[TaskMetaSchema]:
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
    # Typeahead repeats heavily — cache by (filter_key, name).
    return _cached_json(request, _filtered_task_schemas_bytes_named(filter_key, name))


@functools.lru_cache(maxsize=256)
def _filtered_task_schemas_bytes_named(
    filter_key: tuple[
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
        tuple[str, ...] | None,
    ],
    name: str,
) -> Serialized:
    return _serialize_schemas(
        [s for s in _filtered_task_schemas(*filter_key) if name in s.name],
        _TASK_LIST_ADAPTER,
    )


@router.get("/tasks/{name:path}/scores", response_model=TaskScoresSchema)
async def task_scores(request: Request, name: str) -> Response:
    """Per-model scores on task ``name`` across every benchmark that hosts it."""
    _require_task(name)
    return _cached_json(request, await get_task_scores_bytes(name))


@functools.cache
def _descriptive_stats_bytes(name: str) -> Serialized | None:
    """Per-split descriptive-stats JSON for task ``name``; ``None`` if no file on disk."""
    stat_path = _TASKS_REGISTRY[name].metadata.descriptive_stat_path
    if not stat_path.exists():
        return None
    return serialize_bytes(stat_path.read_bytes())


@router.get(
    "/tasks/{name:path}/descriptive_statistics",
    response_model=dict[str, DescriptiveStatsValue],
)
async def task_descriptive_stats(request: Request, name: str) -> Response:
    """Per-split descriptive statistics; shape varies by ``task.type`` (one of ``*DescriptiveStatistics``)."""
    _require_task(name)
    payload = _descriptive_stats_bytes(name)
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"No descriptive stats for task: {name}"
        )
    return _cached_json(request, payload)


@router.get("/tasks/{name:path}", response_model=TaskMetaSchema)
async def task_detail(name: str) -> TaskMetaSchema:
    """Static metadata for task ``name``."""
    _require_task(name)
    TASK_SELECTIONS.labels(name=name).inc()
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
async def list_models(  # noqa: PLR0913, PLR0917
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
    return _cached_json(request, _filtered_model_schemas_bytes_named(filter_key, name))


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
    name: str,
) -> Serialized:
    return _serialize_schemas(
        [s for s in _filtered_model_schemas(*filter_key) if name in s.name],
        _MODEL_LIST_ADAPTER,
    )


@router.get("/models/{name:path}/scores", response_model=ModelScoresSchema)
async def model_scores(request: Request, name: str) -> Response:
    """Per-benchmark scores for model ``name``."""
    _require_model(name)
    return _cached_json(request, await get_model_scores_bytes(name))


@router.get("/models/{name:path}", response_model=ModelMetaSchema)
async def model_detail(name: str) -> ModelMetaSchema:
    """Static metadata for model ``name``."""
    _require_model(name)
    MODEL_SELECTIONS.labels(name=name).inc()
    return model_meta_to_schema(get_model_meta(name))
