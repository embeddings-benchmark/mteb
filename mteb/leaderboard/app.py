from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import tempfile
import time
import warnings
from typing import Literal, get_args
from urllib.parse import urlencode

import cachetools
import gradio as gr
import pandas as pd
import polars as pl

import mteb
from mteb.benchmarks._create_table import _is_zero_shot_cached
from mteb.benchmarks._leaderboard_menu import (
    GP_BENCHMARK_ENTRIES,
    R_BENCHMARK_ENTRIES,
    MenuEntry,
)
from mteb.benchmarks.benchmark import Benchmark, RtebBenchmark
from mteb.cache import ResultCache
from mteb.get_tasks import _TASKS_REGISTRY
from mteb.leaderboard.benchmark_selector import (
    DEFAULT_BENCHMARK_NAME,
    _make_selector,
)
from mteb.leaderboard.event_logger import EventLogger
from mteb.leaderboard.figures import (
    _performance_over_time_plot,
    _performance_size_plot,
    _radar_chart,
)
from mteb.leaderboard.table import (
    apply_per_language_styling_from_benchmark,
    apply_per_task_styling_from_benchmark,
    apply_summary_styling_from_benchmark,
)
from mteb.leaderboard.text_segments import ACKNOWLEDGEMENT, FAQ
from mteb.models.model_meta import MODEL_TYPES
from mteb.results.benchmark_results import BenchmarkResults

logger = logging.getLogger(__name__)
event_logger = EventLogger()


LANGUAGE: list[str] = list({l for t in mteb.get_tasks() for l in t.metadata.languages})
MODEL_TYPE_CHOICES = list(get_args(MODEL_TYPES))


def _produce_benchmark_link(benchmark_name: str, request: gr.Request) -> str:
    """Produces a URL for the selected benchmark.

    Returns:
        A markdown string containing the URL.
    """
    params = urlencode(
        {
            "benchmark_name": benchmark_name,
        }
    )
    base_url = request.request.base_url
    md = "You can also share this benchmark using the following link:\n"
    url = f"{base_url}?{params}"
    md += f"```\n{url}\n```"
    return md


@functools.cache
def _leaderboard_benchmark_names() -> frozenset[str]:
    names: set[str] = set()
    pending: list[Benchmark | MenuEntry] = [
        *GP_BENCHMARK_ENTRIES,
        *R_BENCHMARK_ENTRIES,
    ]

    while pending:
        entry = pending.pop()
        if isinstance(entry, Benchmark):
            names.add(entry.name)
        else:
            pending.extend(entry.benchmarks)

    return frozenset(names)


def _resolve_benchmark_name_from_query(benchmark_name: str | None) -> str:
    if benchmark_name is None:
        return DEFAULT_BENCHMARK_NAME

    try:
        resolved_name = mteb.get_benchmark(benchmark_name).name
    except KeyError:
        return DEFAULT_BENCHMARK_NAME

    if resolved_name not in _leaderboard_benchmark_names():
        return DEFAULT_BENCHMARK_NAME

    return resolved_name


def _set_benchmark_on_load(request: gr.Request) -> str:
    query_params = request.query_params
    return _resolve_benchmark_name_from_query(query_params.get("benchmark_name"))


def _download_table(table: pd.DataFrame) -> str:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    table.to_csv(file)
    return file.name


def _update_citation(benchmark_name: str) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    if benchmark.citation is not None:
        citation = "To cite this work, please use the following reference:\n"
        citation += f"```bibtex\n{benchmark.citation}\n```"
    else:
        citation = ""
    return citation


def _update_description(
    benchmark_name: str, languages: list[str], task_types: list[str], domains: list[str]
) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    description = f"{benchmark.description}\n"
    n_languages = len(languages)
    n_task_types = len(task_types)
    n_tasks = len(benchmark.tasks)
    n_domains = len(domains)
    description += f" - **Number of languages**: {n_languages}\n"
    description += f" - **Number of tasks**: {n_tasks}\n"
    description += f" - **Number of task types**: {n_task_types}\n"
    description += f" - **Number of domains**: {n_domains}\n"
    if benchmark.reference is not None:
        description += (
            f'\n<a href="{benchmark.reference}" target="_blank">Click for More Info</a>'
        )

    return description


def _format_list(props: list[str]):
    if props is None:
        return ""
    if len(props) > 3:
        return ", ".join(props[:3]) + "..."
    return ", ".join(props)


def _update_task_info(task_names: str) -> pd.DataFrame:
    t0 = time.time()
    tasks = mteb.get_tasks(tasks=task_names)
    t1 = time.time()
    df = tasks.to_dataframe(
        properties=[
            "name",
            "type",
            "languages",
            "domains",
            "reference",
            "main_score",
            "modalities",
            "is_public",
        ]
    )
    t2 = time.time()
    df["languages"] = df["languages"].map(_format_list)
    df = df.sort_values("name")
    df["domains"] = df["domains"].map(_format_list)
    df["name"] = df.apply(
        lambda row: f'<a href="{row["reference"]}" target="_blank">{row["name"]}</a>',
        axis=1,
    )
    df["modalities"] = df["modalities"].map(_format_list)
    df = df.rename(
        columns={
            "name": "Task Name",
            "type": "Task Type",
            "languages": "Languages",
            "domains": "Domains",
            "main_score": "Metric",
            "modalities": "Modality",
            "is_public": "Public",
        }
    )
    df = df.drop(columns="reference")
    t3 = time.time()
    logger.info(
        "_update_task_info: get_tasks=%.3fs to_dataframe=%.3fs transform=%.3fs total=%.3fs n_tasks=%d",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t3 - t0,
        len(df.index),
    )
    return df


# Model sizes in million parameters
MIN_MODEL_SIZE, MAX_MODEL_SIZE = 0, 100_000


def _filter_models(
    model_names: list[str],
    task_select: list[str],
    availability: bool | None,
    compatibility: list[str],
    instructions: bool | None,
    max_model_size: int,
    zero_shot_setting: Literal["only_zero_shot", "allow_all", "remove_unknown"],
    model_types: list[str] | None,
):
    lower, upper = 0, max_model_size
    # Setting to None, when the user doesn't specify anything
    if (lower == MIN_MODEL_SIZE) or (lower is None):
        lower = None
    else:
        # Multiplying by millions
        lower = lower * 1e6  # noqa: PLR6104
    if (upper == MAX_MODEL_SIZE) or (upper is None):
        upper = None
    else:
        upper = upper * 1e6  # noqa: PLR6104
    model_metas = mteb.get_model_metas(
        model_names=model_names,
        open_weights=availability,
        use_instructions=instructions,
        frameworks=compatibility,
        n_parameters_range=(lower, upper),
        model_types=model_types,
    )

    # In "allow_all" mode the zero-shot status never filters anything, so skip the
    # (expensive) per-model is_zero_shot_on() traversal entirely.
    if zero_shot_setting == "allow_all":
        return list({model_meta.name for model_meta in model_metas})

    task_name_set = set(task_select) if task_select else set()
    models_to_keep = set()
    for model_meta in model_metas:
        is_model_zero_shot = _is_zero_shot_cached(model_meta.name, task_name_set)
        if is_model_zero_shot is None:
            if zero_shot_setting in ["remove_unknown", "only_zero_shot"]:  # noqa: PLR6201
                continue
        elif not is_model_zero_shot:
            if zero_shot_setting == "only_zero_shot":
                continue
        models_to_keep.add(model_meta.name)
    return list(models_to_keep)


def _should_show_zero_shot_filter(benchmark_name: str) -> bool:
    benchmark = mteb.get_benchmark(benchmark_name)

    if isinstance(benchmark, RtebBenchmark):
        return False
    return True


@cachetools.cached(
    cache={},
    key=lambda benchmark_name, all_benchmark_results: hash(benchmark_name),
)
def _cache_on_benchmark_select(benchmark_name, all_benchmark_results):
    start_time = time.time()
    benchmark = mteb.get_benchmark(benchmark_name)
    languages = [task.languages for task in benchmark.tasks if task.languages]
    languages = set(itertools.chain.from_iterable(languages))
    languages = sorted(languages)
    domains = [
        task.metadata.domains for task in benchmark.tasks if task.metadata.domains
    ]
    domains = set(itertools.chain.from_iterable(domains))
    types = {task.metadata.type for task in benchmark.tasks if task.metadata.type}
    modalities = set()
    for task in benchmark.tasks:
        modalities.update(task.metadata.modalities)
    languages, domains, types, modalities = (
        sorted(languages),
        sorted(domains),
        sorted(types),
        sorted(modalities),
    )
    elapsed = time.time() - start_time
    bm_pl_df = all_benchmark_results[benchmark_name]
    scores = (
        bm_pl_df.select("model_name", "task_name").unique()
        if not bm_pl_df.is_empty() and "model_name" in bm_pl_df.columns
        else _EMPTY_SCORES_FRAME
    )
    logger.debug(f"on_benchmark_select callback: {elapsed}s")
    show_zero_shot = _should_show_zero_shot_filter(benchmark_name)

    # Calculate initial models for this benchmark to avoid race conditions
    benchmark_tasks = sorted([task.metadata.name for task in benchmark.tasks])
    all_models_in_scores = scores["model_name"].unique().to_list()
    initial_models = sorted(
        _filter_models(
            all_models_in_scores,
            benchmark_tasks,
            availability=None,
            compatibility=[],
            instructions=None,
            max_model_size=MAX_MODEL_SIZE,
            zero_shot_setting="allow_all",
            model_types=MODEL_TYPE_CHOICES,
        )
    )

    return (
        languages,
        domains,
        types,
        modalities,
        benchmark_tasks,
        scores,
        show_zero_shot,
        initial_models,
    )


@cachetools.cached(
    cache={},
    key=lambda benchmark_name,
    type_select,
    domain_select,
    lang_select,
    modality_select: (
        hash(
            (
                hash(benchmark_name),
                hash(tuple(type_select)),
                hash(tuple(domain_select)),
                hash(tuple(lang_select)),
                hash(tuple(modality_select)),
            )
        )
    ),
)
def _cache_update_task_list(
    benchmark_name, type_select, domain_select, lang_select, modality_select
):
    if not len(lang_select):
        return []
    start_time = time.time()
    benchmark_tasks = []
    tasks_to_keep = []
    for task in mteb.get_benchmark(benchmark_name).tasks:
        benchmark_tasks.append(task.metadata.name)
        if task.metadata.type not in type_select:
            continue
        if task.metadata.domains and not (
            set(task.metadata.domains) & set(domain_select)
        ):
            continue
        if task.languages and not (set(task.languages) & set(lang_select)):
            continue
        if task.metadata.modalities and not (
            set(task.metadata.modalities) & set(modality_select)
        ):
            continue
        tasks_to_keep.append(task.metadata.name)
    benchmark_tasks.sort()
    tasks_to_keep.sort()
    elapsed = time.time() - start_time
    logger.debug(f"update_task_list callback: {elapsed}s")

    return benchmark_tasks, tasks_to_keep


@functools.lru_cache(maxsize=256)
def _benchmark_full_languages(benchmark_name: str) -> frozenset[str]:
    """All languages covered by a benchmark's tasks (3-letter ISO codes)."""
    benchmark = mteb.get_benchmark(benchmark_name)
    langs: set[str] = set()
    for task in benchmark.tasks:
        if task.languages:
            langs.update(task.languages)
    return frozenset(langs)


_EMPTY_SCORES_FRAME = pl.DataFrame(schema={"model_name": pl.Utf8, "task_name": pl.Utf8})


def _get_session_id(request: gr.Request) -> str:
    """Derive a stable session ID from Gradio's built-in session hash.

    Uses request.session_hash which is always available in every Gradio callback,
    eliminating any dependency on demo.load firing reliably.
    """
    return f"session_{request.session_hash}"


def _get_client_ip(request: gr.Request) -> str:
    """Common proxy/CDN headers. X-Forwarded-For may contain:

    client, proxy1, proxy2
    """
    headers = request.headers

    # Common proxy/CDN headers. X-Forwarded-For may contain:
    # client, proxy1, proxy2
    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    cf_ip = headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()

    return request.client.host if request.client else "unknown"


def _get_visitor_id(request: gr.Request) -> str:
    """Derive a cross-session visitor fingerprint from HTTP request headers.

    Hashes IP + User-Agent + Accept-Language so the same browser gets the same
    visitor_id across separate visits, enabling DAU-style analysis.
    """
    ip = _get_client_ip(request)
    fingerprint = "|".join(
        [
            ip,
            request.headers.get("user-agent", ""),
            request.headers.get("accept-language", ""),
        ]
    )
    return "visitor_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def on_page_load(request: gr.Request):
    """Log page view and collect browser HTTP headers on session start."""
    event_logger.log_page_view(
        session_id=_get_session_id(request),
        properties={
            "visitor_id": _get_visitor_id(request),
            "user_agent": request.headers.get("user-agent", ""),
            "accept_language": request.headers.get("accept-language", ""),
            "referer": request.headers.get("referer", ""),
            "ip": _get_client_ip(request),
        },
    )


def get_leaderboard_app(  # noqa: PLR0914
    cache: ResultCache = ResultCache(),
    rebuild: bool = False,
    cache_repo_id: str = "mteb/results",
) -> gr.Blocks:
    """Returns a Gradio Blocks app for the MTEB leaderboard.

    Args:
        cache: ResultCache instance for managing benchmark data
        rebuild: If True, bypasses the processed parquet cache and rebuilds the
                per-benchmark leaderboard data from results, then rewrites the parquet
                cache. When False (default), startup loads the per-benchmark polars
                frames directly from the parquet cache if present (skipping the results
                load + per-benchmark processing); otherwise it builds them and saves the
                parquet for next time. Use rebuild=True to pick up newly added results.
        cache_repo_id: Optional HF dataset repo id (e.g. "mteb/results"). When set and
                rebuild/push_cache are False, startup loads the parquet cache from the hub
                (``hf://datasets/{cache_repo_id}/leaderboard/benchmark_results.parquet``)
                before falling back to the local parquet cache or a full rebuild.

    Returns:
        gr.Blocks: A Gradio Blocks application configured with the MTEB leaderboard interface
    """
    # Ensure leaderboard timing logs are visible regardless of the caller's
    # logging configuration (e.g. Gradio/uvicorn may leave the root logger at WARNING).
    _lb_logger = logging.getLogger("mteb.leaderboard")
    _lb_logger.setLevel(logging.INFO)
    if not _lb_logger.handlers and not logging.root.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        _lb_logger.addHandler(_handler)

    app_start = time.time()
    logger.info("=== Starting leaderboard app initialization ===")

    logger.info("Step 1/6: Fetching benchmarks...")
    bench_start = time.time()

    seen: set[str] = set()
    benchmarks: list[Benchmark] = []
    pending: list[Benchmark | MenuEntry] = [
        *GP_BENCHMARK_ENTRIES,
        *R_BENCHMARK_ENTRIES,
    ]
    while pending:
        entry = pending.pop()
        if isinstance(entry, Benchmark):
            if entry.name not in seen:
                seen.add(entry.name)
                benchmarks.append(entry)
        else:
            pending.extend(entry.benchmarks)
    benchmarks.sort(key=lambda x: x.name)
    bench_time = time.time() - bench_start
    logger.info(
        f"Step 1/6 complete: Fetched {len(benchmarks)} benchmarks in {bench_time:.2f}s"
    )

    logger.info("Step 2/6: Loading benchmark results...")
    load_start = time.time()
    parquet_path = cache.leaderboard_parquet_path
    loaded: dict[str, pl.DataFrame] | None = None
    use_cache = not rebuild

    def _try_load(source_label: str, loader) -> dict[str, pl.DataFrame] | None:
        try:
            cached = loader()
        except Exception as e:
            logger.warning(f"Failed to load leaderboard cache from {source_label}: {e}")
            return None
        if not cached:
            logger.info(f"{source_label} cache is empty/outdated; trying next source.")
            return None
        logger.info(
            f"Step 2/6 complete: Loaded {len(cached)} benchmarks from "
            f"{source_label} in {time.time() - load_start:.2f}s"
        )
        return cached

    if use_cache and cache_repo_id:
        loaded = _try_load(
            f"hub '{cache_repo_id}'",
            lambda: BenchmarkResults.load_leaderboard_cache(
                cache_repo_id,
                from_hub=True,
            ),
        )
    if loaded is None and use_cache and parquet_path.exists():
        loaded = _try_load(
            f"local parquet {parquet_path}",
            lambda: BenchmarkResults.load_leaderboard_cache(parquet_path),
        )

    if loaded is None:
        all_results = cache._load_from_cache(rebuild=True)
        # Per-benchmark validated frames (scores filtered to each benchmark's task config).
        loaded = {b.name: all_results._to_results_df(b.tasks) for b in benchmarks}
        BenchmarkResults.save_leaderboard_cache(loaded, parquet_path)
        logger.info(
            f"Step 2/6 complete: Built {len(benchmarks)} benchmarks and saved "
            f"{parquet_path} in {time.time() - load_start:.2f}s"
        )

    # Benchmarks absent from the cache (e.g. empty results) become empty frames.
    all_benchmark_results: dict[str, pl.DataFrame] = {
        b.name: loaded.get(b.name, pl.DataFrame()) for b in benchmarks
    }

    default_benchmark = mteb.get_benchmark(DEFAULT_BENCHMARK_NAME)
    default_pl_df = all_benchmark_results[default_benchmark.name]
    default_task_types = {
        _TASKS_REGISTRY[t].metadata.type
        for t in default_pl_df["task_name"].unique().to_list()
        if t in _TASKS_REGISTRY
        and _TASKS_REGISTRY[t].metadata.type != "InstructionRetrieval"
    }
    display_radar_chart = len(default_task_types) > 1

    logger.info("Step 3/6: Filtering models...")
    filter_start = time.time()

    default_scores = default_pl_df.select("model_name", "task_name").unique()
    all_models = default_pl_df["model_name"].unique().to_list()
    default_task_names_for_filter = sorted(
        default_pl_df["task_name"].unique().to_list()
    )
    filtered_models = _filter_models(
        all_models,
        default_task_names_for_filter,
        availability=None,
        compatibility=[],
        instructions=None,
        max_model_size=MAX_MODEL_SIZE,
        zero_shot_setting="allow_all",
        model_types=MODEL_TYPE_CHOICES,
    )
    filtered_model_names = (
        default_scores.filter(pl.col("model_name").is_in(set(filtered_models)))[
            "model_name"
        ]
        .unique()
        .to_list()
    )
    filtered_pl_df = default_pl_df.filter(
        pl.col("model_name").is_in(filtered_model_names)
    )
    filter_time = time.time() - filter_start
    logger.info(
        f"Step 3/6 complete: Filtered {len(filtered_model_names)} models in {filter_time:.2f}s"
    )

    logger.info("Step 4/6: Generating tables...")
    table_start = time.time()
    summary_table, summary_raw = apply_summary_styling_from_benchmark(
        default_benchmark, filtered_pl_df
    )
    per_task_table = apply_per_task_styling_from_benchmark(
        default_benchmark, filtered_pl_df
    )
    per_language_table = apply_per_language_styling_from_benchmark(
        default_benchmark,
        filtered_pl_df,
    )
    table_time = time.time() - table_start
    logger.info(f"Step 4/6 complete: Generated tables in {table_time:.2f}s")

    display_language_table = len(default_benchmark.language_view) > 0

    logger.info("Step 5/6: Creating Gradio components...")
    component_start = time.time()
    default_languages = sorted(_benchmark_full_languages(default_benchmark.name))
    default_task_types = sorted(
        {
            _TASKS_REGISTRY[t].metadata.type
            for t in default_pl_df["task_name"].unique().to_list()
            if t in _TASKS_REGISTRY
        }
    )
    default_domains = sorted(
        {
            d
            for t in default_pl_df["task_name"].unique().to_list()
            if t in _TASKS_REGISTRY
            for d in (_TASKS_REGISTRY[t].metadata.domains or [])
        }
    )
    default_task_names = sorted(default_pl_df["task_name"].unique().to_list())
    default_modalities = sorted(
        {
            m
            for t in default_pl_df["task_name"].unique().to_list()
            if t in _TASKS_REGISTRY
            for m in (_TASKS_REGISTRY[t].metadata.modalities or [])
        }
    )
    lang_select = gr.CheckboxGroup(
        default_languages,
        value=default_languages,
        show_label=True,
        show_select_all=True,
        label="Language",
        info="Select languages to include.",
    )
    type_select = gr.CheckboxGroup(
        default_task_types,
        value=default_task_types,
        show_label=True,
        show_select_all=True,
        label="Task Type",
        info="Select task types to include.",
    )
    domain_select = gr.CheckboxGroup(
        default_domains,
        value=default_domains,
        show_label=True,
        show_select_all=True,
        label="Domain",
        info="Select domains to include.",
    )
    task_select = gr.CheckboxGroup(
        default_task_names,
        value=default_task_names,
        show_label=True,
        show_select_all=True,
        label="Task",
        info="Select specific tasks to include",
    )
    modality_select = gr.CheckboxGroup(
        default_modalities,
        value=default_modalities,
        show_label=True,
        show_select_all=True,
        label="Modality",
        info="Select modalities to include.",
    )
    component_time = time.time() - component_start
    logger.info(
        f"Step 5/6 complete: Created Gradio components in {component_time:.2f}s"
    )

    logger.info("Step 6/6: Building Gradio interface and callbacks...")
    interface_start = time.time()
    with gr.Blocks(  # noqa: PLR1702
        title="MTEB Leaderboard",
        fill_width=True,
    ) as demo:
        demo.load(fn=on_page_load)

        with gr.Sidebar(
            position="left",
            label="Benchmark Selection and Customization",
            visible=True,
            width="18%",
        ):
            benchmark_select, column = _make_selector(
                GP_BENCHMARK_ENTRIES + R_BENCHMARK_ENTRIES
            )

        gr.Markdown(
            """
        ## Embedding Leaderboard

        This leaderboard compares 100+ text and image embedding models across 1000+ languages. We refer to the publication of each selectable benchmark for details on metrics, languages, tasks, and task types. Anyone is welcome [to add a model](https://embeddings-benchmark.github.io/mteb/contributing/adding_a_model/), [add benchmarks](https://embeddings-benchmark.github.io/mteb/contributing/adding_a_benchmark/), [help us improve zero-shot annotations](https://github.com/embeddings-benchmark/mteb/blob/06489abca007261c7e6b11f36d4844c5ed5efdcb/mteb/models/bge_models.py#L91) or [propose other changes to the leaderboard](https://github.com/embeddings-benchmark/mteb/issues/new?template=enhancement.yaml).
        """
        )
        gr.Markdown(
            lambda name: f"<center> <h2> <b> {name} </b> </h2> </center><br>",
            inputs=benchmark_select,
        )

        scores = gr.State(default_scores)
        models = gr.State(filtered_models)
        summary_data = gr.State(summary_raw)
        with gr.Row():
            with gr.Column(scale=1):
                description = gr.Markdown(
                    _update_description(
                        default_benchmark.name,
                        default_languages,
                        default_task_types,
                        default_domains,
                    )
                )

            with gr.Column(scale=1):
                with gr.Accordion("Cite and share this benchmark", open=False):
                    citation = gr.Markdown(_update_citation, inputs=[benchmark_select])  # noqa: F841
                    gr.Markdown(_produce_benchmark_link, inputs=[benchmark_select])

                with gr.Accordion(
                    "Customize this Benchmark",
                    open=False,
                ):
                    with gr.Column():
                        with gr.Row():
                            type_select.render()
                        with gr.Row():
                            domain_select.render()
                        with gr.Row():
                            modality_select.render()
                        with gr.Row(elem_classes="overflow-y-scroll max-h-80"):
                            lang_select.render()
                        with gr.Row(elem_classes="overflow-y-scroll max-h-80"):
                            task_select.render()

                with gr.Accordion("Advanced Model Filters", open=False):
                    with gr.Group():
                        with gr.Row(elem_classes=""):
                            with gr.Column():
                                compatibility = gr.CheckboxGroup(
                                    [
                                        (
                                            "Should be sentence-transformers compatible",
                                            "Sentence Transformers",
                                        )
                                    ],
                                    value=[],
                                    label="Compatibility",
                                    interactive=True,
                                )
                                availability = gr.Radio(
                                    [
                                        ("Only Open", True),
                                        ("Only Proprietary", False),
                                        ("Both", None),
                                    ],
                                    value=None,
                                    label="Availability",
                                    interactive=True,
                                )
                                instructions = gr.Radio(
                                    [
                                        ("Only Instruction-tuned", True),
                                        ("Only non-instruction", False),
                                        ("Both", None),
                                    ],
                                    value=None,
                                    label="Instructions",
                                    interactive=True,
                                )
                            with gr.Column():
                                zero_shot = gr.Radio(
                                    [
                                        (
                                            "Only Zero-shot",
                                            "only_zero_shot",
                                        ),
                                        ("Remove Unknown", "remove_unknown"),
                                        ("Allow All", "allow_all"),
                                    ],
                                    value="allow_all",
                                    label="Zero-shot",
                                    interactive=True,
                                )

                                max_model_size = gr.Radio(
                                    [
                                        ("<100M", 100),
                                        ("<500M", 500),
                                        ("<1B", 1000),
                                        ("<5B", 5000),
                                        ("<10B", 10000),
                                        (">10B", MAX_MODEL_SIZE),
                                    ],
                                    value=MAX_MODEL_SIZE,
                                    label="Model Parameters",
                                    interactive=True,
                                )
                            with gr.Column():
                                model_type_select = gr.CheckboxGroup(
                                    MODEL_TYPE_CHOICES,
                                    value=MODEL_TYPE_CHOICES,
                                    label="Model Type",
                                )

        with gr.Tab("Summary"):
            summary_table.render()
            download_summary = gr.DownloadButton("Download Table")
            download_summary.click(
                _download_table, inputs=[summary_table], outputs=[download_summary]
            )

            with gr.Accordion(
                "Frequently Asked Questions",
                open=False,
            ):
                gr.Markdown(FAQ)

        with gr.Tab("Performance per Model Size") as plot_tab:
            plot = gr.Plot(_performance_size_plot, inputs=[summary_data])
            plot_tab.select(
                _performance_size_plot, inputs=[summary_data], outputs=[plot]
            )

        with gr.Tab("Performance over Time") as timeline_tab:
            timeline_plot = gr.Plot(_performance_over_time_plot, inputs=[summary_data])
            timeline_tab.select(
                _performance_over_time_plot,
                inputs=[summary_data],
                outputs=[timeline_plot],
            )

        with gr.Tab(
            "Performance per Task Type", visible=display_radar_chart
        ) as radar_plot_tab:
            radar_plot = gr.Plot(_radar_chart, inputs=[summary_data])
            gr.Markdown(
                "*We only display TOP 5 models that have been run on all task types in the benchmark*"
            )
            radar_plot_tab.select(
                _radar_chart, inputs=[summary_data], outputs=[radar_plot]
            )

        with gr.Tab("Performance per task"):
            per_task_table.render()
            download_per_task = gr.DownloadButton("Download Table")
            download_per_task.click(
                _download_table, inputs=[per_task_table], outputs=[download_per_task]
            )
        with gr.Tab(
            "Performance per language", visible=display_language_table
        ) as language_tab:
            per_language_table.render()
            download_per_language = gr.DownloadButton("Download Table")
            download_per_language.click(
                _download_table,
                inputs=[per_language_table],
                outputs=[download_per_language],
            )
        with gr.Tab("Task information"):
            task_info_table = gr.DataFrame(
                _update_task_info(default_task_names),
                datatype=["markdown"] + ["str"] * 6,
                buttons=["copy", "fullscreen"],
                show_search="filter",
            )

        # This sets the benchmark from the URL query parameters
        demo.load(_set_benchmark_on_load, inputs=[], outputs=[benchmark_select])

        def _estimate_payload_size(obj) -> int:
            """Rough byte estimate of a single output value for telemetry only.

            Uses sys.getsizeof for primitives and len() on str/bytes-like; for
            other objects we stringify them. Not exact, but enough to spot
            multi-MB returns that hurt browser rendering.
            """
            import sys

            try:
                if obj is None:
                    return 0
                if isinstance(obj, (str, bytes)):
                    return len(obj)
                if isinstance(obj, (list, tuple, dict, set)):
                    return sys.getsizeof(obj)
                if isinstance(obj, pd.DataFrame):
                    return int(obj.memory_usage(deep=True).sum())
                return sys.getsizeof(obj)
            except Exception:
                return -1

        def on_benchmark_select(benchmark_name, request: gr.Request | None = None):  # noqa: PLR0914
            t0 = time.time()
            (
                languages,
                domains,
                types,
                modalities,
                benchmark_tasks,
                scores,
                show_zero_shot,
                initial_models,
            ) = _cache_on_benchmark_select(benchmark_name, all_benchmark_results)
            t1 = time.time()

            if request:
                event_logger.log_benchmark_change(
                    session_id=_get_session_id(request),
                    new_value=benchmark_name,
                    old_value=None,
                    properties={"visitor_id": _get_visitor_id(request)},
                )

            bm_pl_df = all_benchmark_results[benchmark_name]
            if bm_pl_df.is_empty() or "task_name" not in bm_pl_df.columns:
                eligible_task_types = set()
            else:
                eligible_task_types = {
                    _TASKS_REGISTRY[t].metadata.type
                    for t in bm_pl_df["task_name"].unique().to_list()
                    if t in _TASKS_REGISTRY
                    and _TASKS_REGISTRY[t].metadata.type != "InstructionRetrieval"
                }
            display_radar = len(eligible_task_types) > 1
            t2 = time.time()
            (
                summary_table_value,
                summary_raw,
                per_task_table_value,
                per_language_table_value,
                language_tab_update,
            ) = update_tables(
                scores,
                benchmark_tasks,
                initial_models,
                benchmark_name,
                languages,
            )
            t3 = time.time()
            size_plot = _performance_size_plot(summary_raw)
            t4 = time.time()
            time_plot = _performance_over_time_plot(summary_raw)
            t5 = time.time()
            radar = _radar_chart(summary_raw)
            t6 = time.time()
            task_info_value = _update_task_info(benchmark_tasks)
            t7 = time.time()
            description_value = _update_description(
                benchmark_name, languages, types, domains
            )
            t8 = time.time()
            outputs = (
                gr.update(choices=languages, value=languages),
                gr.update(choices=domains, value=domains),
                gr.update(choices=types, value=types),
                gr.update(choices=modalities, value=modalities),
                gr.update(choices=benchmark_tasks, value=benchmark_tasks),
                scores,
                gr.update(visible=show_zero_shot),
                initial_models,
                gr.update(visible=display_radar),
                gr.update(value=summary_raw),
                size_plot,
                time_plot,
                radar,
                summary_table_value,
                per_task_table_value,
                per_language_table_value,
                language_tab_update,
                task_info_value,
                description_value,
            )
            output_names = (
                "lang",
                "dom",
                "type",
                "mod",
                "task",
                "scores",
                "zs",
                "models",
                "radar_tab",
                "summary_data",
                "size_plot",
                "time_plot",
                "radar",
                "summary_tbl",
                "per_task_tbl",
                "per_lang_tbl",
                "lang_tab",
                "task_info",
                "desc",
            )
            sizes = {
                n: _estimate_payload_size(o) for n, o in zip(output_names, outputs)
            }
            total_size = sum(s for s in sizes.values() if s > 0)
            t9 = time.time()
            logger.info(
                "on_benchmark_select [%s]: cache=%.3fs task_types=%.3fs tables=%.3fs size_plot=%.3fs time_plot=%.3fs radar=%.3fs task_info=%.3fs desc=%.3fs size_est=%.3fs total=%.3fs payload=%dKB sizes=%s",
                benchmark_name,
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
                t6 - t5,
                t7 - t6,
                t8 - t7,
                t9 - t8,
                t9 - t0,
                total_size // 1024,
                {
                    n: f"{s // 1024}KB"
                    for n, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:6]
                },
            )
            return outputs

        benchmark_select.change(
            on_benchmark_select,
            inputs=[benchmark_select],
            outputs=[
                lang_select,
                domain_select,
                type_select,
                modality_select,
                task_select,
                scores,
                zero_shot,
                models,
                radar_plot_tab,
                summary_data,
                plot,
                timeline_plot,
                radar_plot,
                summary_table,
                per_task_table,
                per_language_table,
                language_tab,
                task_info_table,
                description,
            ],
        )

        @cachetools.cached(
            cache={},
            key=lambda benchmark_name, languages: hash(
                (hash(benchmark_name), hash(tuple(languages)))
            ),
        )
        def update_scores_on_lang_change(benchmark_name, languages):
            start_time = time.time()
            if not len(languages):
                return _EMPTY_SCORES_FRAME
            # lang_select stores 3-letter ISO codes ("eng") while the language
            # column stores full language-script codes ("eng-Latn"). Strip the
            # script suffix before comparing.
            lang_set = set(languages)
            scores = (
                all_benchmark_results[benchmark_name]
                .filter(
                    pl.col("language")
                    .list.eval(pl.element().str.split("-").list.first().is_in(lang_set))
                    .list.any()
                )
                .select("model_name", "task_name")
                .unique()
            )
            elapsed = time.time() - start_time
            logger.debug(f"update_scores callback: {elapsed}s")
            return scores

        def update_task_list(
            benchmark_name,
            type_select,
            domain_select,
            lang_select,
            modality_select,
            request: gr.Request | None = None,
        ):
            benchmark_tasks, tasks_to_keep = _cache_update_task_list(
                benchmark_name, type_select, domain_select, lang_select, modality_select
            )
            if request:
                event_logger.log_filter_change(
                    session_id=_get_session_id(request),
                    filter_name="task_type",
                    new_value=benchmark_name,
                    old_value=None,
                    benchmark=benchmark_name,
                    filters={
                        "task_type": type_select,
                        "domain": domain_select,
                        "language": lang_select,
                        "modality": modality_select,
                    },
                    properties={"visitor_id": _get_visitor_id(request)},
                )
            return gr.update(choices=benchmark_tasks, value=tasks_to_keep)

        @cachetools.cached(
            cache={},
            key=lambda scores,
            tasks,
            availability,
            compatibility,
            instructions,
            max_model_size,
            zero_shot,
            model_type_select,
            request=None: (
                hash(
                    (
                        id(scores),
                        hash(tuple(tasks)),
                        hash(availability),
                        hash(tuple(compatibility)),
                        hash(instructions),
                        hash(max_model_size),
                        hash(zero_shot),
                        hash(tuple(model_type_select)),
                    )
                )
            ),
        )
        def update_models(
            scores: pl.DataFrame,
            tasks: list[str],
            availability: bool | None,
            compatibility: list[str],
            instructions: bool | None,
            max_model_size: int,
            zero_shot: Literal["allow_all", "remove_unknown", "only_zero_shot"],
            model_type_select: list[str],
            request: gr.Request | None = None,
        ):
            start_time = time.time()
            model_names = scores["model_name"].unique().to_list()
            filtered_models = _filter_models(
                model_names,
                tasks,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot_setting=zero_shot,
                model_types=model_type_select,
            )
            elapsed = time.time() - start_time
            logger.debug(f"update_models callback: {elapsed}s")
            if request:
                event_logger.log_filter_change(
                    session_id=_get_session_id(request),
                    filter_name="model_type",
                    new_value=None,
                    old_value=None,
                    benchmark=None,
                    filters={
                        "tasks": tasks,
                        "availability": availability,
                        "compatibility": compatibility,
                        "instructions": instructions,
                        "max_model_size": max_model_size,
                        "zero_shot": zero_shot,
                    },
                    properties={"visitor_id": _get_visitor_id(request)},
                )

            return sorted(filtered_models)

        def _cache_key_for_update_tables(
            scores, tasks, models_to_keep, benchmark_name, languages
        ):
            # `scores` is fully determined by `(benchmark_name, languages)` upstream,
            # so we hash those instead of the O(N log N) sort over the scores list.
            tasks_hash = hash(tuple(sorted(tasks))) if tasks is not None else None
            models_hash = (
                hash(tuple(sorted(models_to_keep)))
                if models_to_keep is not None
                else None
            )
            bench_hash = hash(benchmark_name)
            lang_hash = (
                hash(tuple(sorted(languages))) if languages is not None else None
            )
            return hash((bench_hash, lang_hash, tasks_hash, models_hash))

        _update_tables_cache: dict = {}

        @cachetools.cached(
            cache=_update_tables_cache,
            key=_cache_key_for_update_tables,
        )
        def update_tables(  # noqa: PLR0914
            scores,
            tasks,
            models_to_keep,
            benchmark_name: str,
            languages: list[str],
        ):
            # Reaching the body means cachetools missed — log so unexpected misses surface.
            logger.info(
                "update_tables cache MISS [%s] tasks=%d models=%d langs=%d",
                benchmark_name,
                len(tasks) if tasks is not None else -1,
                len(models_to_keep) if models_to_keep is not None else -1,
                len(languages) if languages is not None else -1,
            )
            start_time = time.time()
            tasks_set = set(tasks) if tasks is not None else None
            benchmark = mteb.get_benchmark(benchmark_name)

            # Restrict the (model, task) pair set to current UI selections in polars,
            # then extract the unique model and task name sets the polars filter below
            # will use as ``is_in`` predicates on the full pre-aggregation frame.
            scored = scores
            if tasks_set is not None:
                scored = scored.filter(pl.col("task_name").is_in(tasks_set))
            if models_to_keep is not None:
                scored = scored.filter(pl.col("model_name").is_in(set(models_to_keep)))
            filtered_model_names = set(scored["model_name"].unique().to_list())
            filtered_task_names = set(scored["task_name"].unique().to_list())

            t_filter0 = time.time()
            bm_pl_df = all_benchmark_results[benchmark_name]
            if not filtered_task_names or bm_pl_df.is_empty():
                filtered_df = bm_pl_df.clear()
            else:
                mask = pl.col("model_name").is_in(filtered_model_names) & pl.col(
                    "task_name"
                ).is_in(filtered_task_names)
                # Restrict to selected languages (mirrors update_scores_on_lang_change);
                # skipped when the selection covers every benchmark language.
                if languages is not None and not set(languages).issuperset(
                    _benchmark_full_languages(benchmark_name)
                ):
                    lang_set = set(languages)
                    mask &= (
                        pl.col("language")
                        .list.eval(
                            pl.element().str.split("-").list.first().is_in(lang_set)
                        )
                        .list.any()
                    )
                # Streaming engine evaluates the is_in mask on the ~9M-row frame
                # ~4× faster than the default in-memory engine here.
                filtered_df = bm_pl_df.lazy().filter(mask).collect(engine="streaming")
            t_filter1 = time.time()
            summary, summary_raw = apply_summary_styling_from_benchmark(
                benchmark, filtered_df
            )
            t_summary = time.time()
            per_task = apply_per_task_styling_from_benchmark(benchmark, filtered_df)
            t_per_task = time.time()
            per_language = apply_per_language_styling_from_benchmark(
                benchmark,
                filtered_df,
            )
            elapsed = time.time() - start_time
            logger.info(
                "update_tables [%s]: scores_filter=%.3fs pl_filter=%.3fs summary=%.3fs per_task=%.3fs per_language=%.3fs total=%.3fs",
                benchmark_name,
                t_filter0 - start_time,
                t_filter1 - t_filter0,
                t_summary - t_filter1,
                t_per_task - t_summary,
                elapsed - (t_per_task - start_time),
                elapsed,
            )
            return (
                summary,
                summary_raw,
                per_task,
                per_language,
                gr.update(visible=len(benchmark.language_view) > 0),
            )

        # Event wiring. Handlers use .input (not .change) so programmatic gr.update()
        # from on_benchmark_select does not re-fire them.
        _task_list_inputs = [
            benchmark_select,
            type_select,
            domain_select,
            lang_select,
            modality_select,
        ]
        _model_filter_inputs = [
            scores,
            task_select,
            availability,
            compatibility,
            instructions,
            max_model_size,
            zero_shot,
            model_type_select,
        ]
        _table_inputs = [scores, task_select, models, benchmark_select, lang_select]
        _table_outputs = [
            summary_table,
            summary_data,
            per_task_table,
            per_language_table,
            language_tab,
        ]

        # Description updates from user changes to language/type/domain filters.
        gr.on(
            triggers=[lang_select.input, type_select.input, domain_select.input],
            fn=_update_description,
            inputs=[benchmark_select, lang_select, type_select, domain_select],
            outputs=[description],
            show_progress="hidden",
        )

        # Language filter chain: scores -> task list -> task info -> models -> tables.
        lang_select.input(
            update_scores_on_lang_change,
            inputs=[benchmark_select, lang_select],
            outputs=[scores],
        ).then(
            update_task_list,
            inputs=_task_list_inputs,
            outputs=[task_select],
        ).then(
            _update_task_info,
            inputs=[task_select],
            outputs=[task_info_table],
        ).then(
            update_models,
            inputs=_model_filter_inputs,
            outputs=[models],
        ).then(
            update_tables,
            inputs=_table_inputs,
            outputs=_table_outputs,
        )

        # Type / domain / modality chains: task list -> task info -> models -> tables.
        gr.on(
            triggers=[type_select.input, domain_select.input, modality_select.input],
            fn=update_task_list,
            inputs=_task_list_inputs,
            outputs=[task_select],
        ).then(
            _update_task_info,
            inputs=[task_select],
            outputs=[task_info_table],
        ).then(
            update_models,
            inputs=_model_filter_inputs,
            outputs=[models],
        ).then(
            update_tables,
            inputs=_table_inputs,
            outputs=_table_outputs,
        )

        # Direct task selection by user: task info -> models -> tables.
        task_select.input(
            _update_task_info,
            inputs=[task_select],
            outputs=[task_info_table],
        ).then(
            update_models,
            inputs=_model_filter_inputs,
            outputs=[models],
        ).then(
            update_tables,
            inputs=_table_inputs,
            outputs=_table_outputs,
        )

        # Model-filter inputs: models -> tables.
        gr.on(
            triggers=[
                availability.input,
                compatibility.input,
                instructions.input,
                max_model_size.input,
                zero_shot.input,
                model_type_select.input,
            ],
            fn=update_models,
            inputs=_model_filter_inputs,
            outputs=[models],
        ).then(
            update_tables,
            inputs=_table_inputs,
            outputs=_table_outputs,
        )

        gr.Markdown(ACKNOWLEDGEMENT, elem_id="ack_markdown")
    interface_time = time.time() - interface_start
    logger.info(f"Step 6/6 complete: Built Gradio interface in {interface_time:.2f}s")

    logger.info("Starting prerun on all benchmarks to populate caches...")
    prerun_start = time.time()
    # Prime the initial-state + filtered-tasks cache entries for every benchmark.
    for benchmark in benchmarks:
        (
            bench_languages,
            bench_domains,
            bench_types,
            bench_modalities,
            bench_tasks,
            bench_scores,
            _zero_shot,
            bench_initial_models,
            _display_radar,
            _summary_raw,
            _perf_size_plot,
            _perf_time_plot,
            _radar_chart_plot,
            _summary_table_val,
            _per_task_table_val,
            _per_language_table_val,
            _language_tab_val,
            _task_info_val,
            _description_val,
        ) = on_benchmark_select(benchmark.name)
        _, tasks_to_keep = _cache_update_task_list(
            benchmark.name,
            bench_types,
            bench_domains,
            bench_languages,
            bench_modalities,
        )
        update_tables(
            bench_scores,
            tasks_to_keep,
            bench_initial_models,
            benchmark.name,
            bench_languages,
        )
    prerun_time = time.time() - prerun_start
    logger.info(
        f"Prerun complete: Processed {len(benchmarks)} benchmarks in {prerun_time:.2f}s"
    )

    total_time = time.time() - app_start
    logger.info(f"=== Leaderboard app initialization complete in {total_time:.2f}s ===")
    return demo


if __name__ == "__main__":
    import os

    # Add process ID to logging for multiprocessing debugging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing handlers
    )

    # Flush log handlers immediately (helpful for multiprocessing)
    for handler in logging.root.handlers:
        handler.flush()

    logger.info(f"Starting leaderboard app in process {os.getpid()}")

    # Suppress specific WARNING messages while keeping INFO level for the app
    logging.getLogger("mteb.results.task_result").setLevel(logging.ERROR)
    logging.getLogger("mteb.models.model_meta").setLevel(logging.ERROR)
    logging.getLogger("mteb.results.benchmark_results").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message="Couldn't get scores for .* due to .*")
    warnings.filterwarnings("ignore", message="Could not get source model: .*")
    warnings.filterwarnings(
        "ignore", message="No scores data available. Returning empty DataFrame."
    )
    warnings.filterwarnings("ignore", message="Main score .* not found in scores")
    warnings.filterwarnings("ignore", message=".*: Missing subsets .* for split .*")
    warnings.filterwarnings("ignore", message=".*: Missing splits .*")

    app = get_leaderboard_app()

    head = """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    """
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Roboto Mono"), "Arial", "sans-serif"],
        ),
        head=head,
    )
