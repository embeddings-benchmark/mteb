import itertools
import json
import logging
import tempfile
import time
import warnings
from pathlib import Path
from typing import Literal, get_args
from urllib.parse import urlencode

import cachetools
import gradio as gr
import pandas as pd

import mteb
from mteb import BenchmarkResults
from mteb.abstasks.task_metadata import TaskDomain, TaskType
from mteb.benchmarks.benchmark import RtebBenchmark
from mteb.cache import ResultCache
from mteb.leaderboard.benchmark_selector import (
    DEFAULT_BENCHMARK_NAME,
    GP_BENCHMARK_ENTRIES,
    R_BENCHMARK_ENTRIES,
    _make_selector,
)
from mteb.leaderboard.figures import _performance_size_plot, _radar_chart
from mteb.leaderboard.table import (
    apply_per_task_styling_from_benchmark,
    apply_summary_styling_from_benchmark,
)
from mteb.leaderboard.text_segments import ACKNOWLEDGEMENT, FAQ
from mteb.types import Modalities

logger = logging.getLogger(__name__)

LANGUAGE: list[str] = list({l for t in mteb.get_tasks() for l in t.metadata.languages})


def _load_results(cache: ResultCache) -> BenchmarkResults:
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        cache.download_from_remote()
        all_model_names = [model_meta.name for model_meta in mteb.get_model_metas()]

        all_results = cache.load_results(
            models=all_model_names,
            only_main_score=True,
            require_model_meta=False,
            include_remote=True,
        )
        return all_results
    else:
        with results_cache_path.open() as cache_file:
            return mteb.BenchmarkResults.from_validated(**json.load(cache_file))


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


def _set_benchmark_on_load(request: gr.Request):
    query_params = request.query_params
    return query_params.get("benchmark_name", DEFAULT_BENCHMARK_NAME)


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


def _update_task_info(task_names: str) -> gr.DataFrame:
    tasks = mteb.get_tasks(tasks=task_names)
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
    df["languages"] = df["languages"].map(_format_list)
    df = df.sort_values("name")
    df["domains"] = df["domains"].map(_format_list)
    df["name"] = f'<a href="{df["reference"]}" target="_blank">{df["name"]}</a>'
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
    return gr.DataFrame(
        df,
        datatype=["markdown"] + ["str"] * (len(df.columns) - 1),
        show_copy_button=True,
        show_fullscreen_button=True,
        show_search="filter",
    )


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
):
    lower, upper = 0, max_model_size
    # Setting to None, when the user doesn't specify anything
    if (lower == MIN_MODEL_SIZE) or (lower is None):
        lower = None
    else:
        # Multiplying by millions
        lower = lower * 1e6
    if (upper == MAX_MODEL_SIZE) or (upper is None):
        upper = None
    else:
        upper = upper * 1e6
    model_metas = mteb.get_model_metas(
        model_names=model_names,
        open_weights=availability,
        use_instructions=instructions,
        frameworks=compatibility,
        n_parameters_range=(lower, upper),
    )

    models_to_keep = set()
    for model_meta in model_metas:
        is_model_zero_shot = model_meta.is_zero_shot_on(task_select)
        if is_model_zero_shot is None:
            if zero_shot_setting in ["remove_unknown", "only_zero_shot"]:
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


def get_leaderboard_app(cache: ResultCache = ResultCache()) -> gr.Blocks:
    """Returns a Gradio Blocks app for the MTEB leaderboard."""
    logger.info("Loading all benchmark results")
    all_results = _load_results(cache)

    benchmarks = sorted(
        mteb.get_benchmarks(display_on_leaderboard=True), key=lambda x: x.name
    )
    all_benchmark_results = {
        benchmark.name: all_results.select_tasks(benchmark.tasks).join_revisions()
        for benchmark in benchmarks
    }
    default_benchmark = mteb.get_benchmark(DEFAULT_BENCHMARK_NAME)
    default_results = all_benchmark_results[default_benchmark.name]
    logger.info("Benchmark results loaded")

    default_scores = default_results._get_scores(format="long")
    all_models = list({entry["model_name"] for entry in default_scores})
    filtered_models = _filter_models(
        all_models,
        default_results.task_names,
        availability=None,
        compatibility=[],
        instructions=None,
        max_model_size=MAX_MODEL_SIZE,
        zero_shot_setting="allow_all",
    )
    default_filtered_scores = [
        entry for entry in default_scores if entry["model_name"] in filtered_models
    ]

    # Filter BenchmarkResults based on default filtered models (as required by Kenneth)
    filtered_model_names = [entry["model_name"] for entry in default_filtered_scores]
    filtered_benchmark_results = default_results.select_models(filtered_model_names)

    summary_table = apply_summary_styling_from_benchmark(
        default_benchmark, filtered_benchmark_results
    )
    per_task_table = apply_per_task_styling_from_benchmark(
        default_benchmark, filtered_benchmark_results
    )

    lang_select = gr.Dropdown(
        LANGUAGE,
        value=sorted(default_results.languages),
        allow_custom_value=True,
        multiselect=True,
        label="Language",
        info="Select languages to include.",
    )
    type_select = gr.Dropdown(
        sorted(get_args(TaskType)),
        value=sorted(default_results.task_types),
        multiselect=True,
        label="Task Type",
        info="Select task types to include.",
    )
    domain_select = gr.Dropdown(
        sorted(get_args(TaskDomain)),
        value=sorted(default_results.domains),
        multiselect=True,
        label="Domain",
        info="Select domains to include.",
    )
    task_select = gr.Dropdown(
        sorted(all_results.task_names),
        value=sorted(default_results.task_names),
        allow_custom_value=True,
        multiselect=True,
        label="Task",
        info="Select specific tasks to include",
    )
    modality_select = gr.Dropdown(
        sorted(get_args(Modalities)),
        value=sorted(default_results.modalities),
        multiselect=True,
        label="Modality",
        info="Select modalities to include.",
    )

    head = """
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    """

    with gr.Blocks(
        fill_width=True,
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Roboto Mono"), "Arial", "sans-serif"],
        ),
        head=head,
    ) as demo:
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
        with gr.Row():
            with gr.Column(scale=1):
                description = gr.Markdown(  # noqa: F841
                    _update_description,
                    inputs=[benchmark_select, lang_select, type_select, domain_select],
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
            plot = gr.Plot(_performance_size_plot, inputs=[summary_table])
            gr.Markdown(
                "*We only display TOP 5 models that have been run on all tasks in the benchmark*"
            )
            plot_tab.select(
                _performance_size_plot, inputs=[summary_table], outputs=[plot]
            )

        with gr.Tab("Performance per Task Type") as radar_plot_tab:
            radar_plot = gr.Plot(_radar_chart, inputs=[summary_table])
            gr.Markdown(
                "*We only display TOP 5 models that have been run on all task types in the benchmark*"
            )
            radar_plot_tab.select(
                _radar_chart, inputs=[summary_table], outputs=[radar_plot]
            )

        with gr.Tab("Performance per task"):
            per_task_table.render()
            download_per_task = gr.DownloadButton("Download Table")
            download_per_task.click(
                _download_table, inputs=[per_task_table], outputs=[download_per_task]
            )
        with gr.Tab("Task information"):
            task_info_table = gr.DataFrame(_update_task_info, inputs=[task_select])  # noqa: F841

        # This sets the benchmark from the URL query parameters
        demo.load(_set_benchmark_on_load, inputs=[], outputs=[benchmark_select])

        @cachetools.cached(
            cache={},
            key=lambda benchmark_name: hash(benchmark_name),
        )
        def on_benchmark_select(benchmark_name):
            start_time = time.time()
            benchmark = mteb.get_benchmark(benchmark_name)
            languages = [task.languages for task in benchmark.tasks if task.languages]
            languages = set(itertools.chain.from_iterable(languages))
            languages = sorted(languages)
            domains = [
                task.metadata.domains
                for task in benchmark.tasks
                if task.metadata.domains
            ]
            domains = set(itertools.chain.from_iterable(domains))
            types = {
                task.metadata.type for task in benchmark.tasks if task.metadata.type
            }
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
            benchmark_results = all_benchmark_results[benchmark_name]
            scores = benchmark_results._get_scores(format="long")
            logger.debug(f"on_benchmark_select callback: {elapsed}s")
            show_zero_shot = _should_show_zero_shot_filter(benchmark_name)

            # Calculate initial models for this benchmark to avoid race conditions
            benchmark_tasks = sorted([task.metadata.name for task in benchmark.tasks])
            all_models_in_scores = list({entry["model_name"] for entry in scores})
            initial_models = _filter_models(
                all_models_in_scores,
                benchmark_tasks,
                availability=None,
                compatibility=[],
                instructions=None,
                max_model_size=MAX_MODEL_SIZE,
                zero_shot_setting="allow_all",
            )
            # Sort to ensure consistency with update_models
            initial_models = sorted(initial_models)

            return (
                languages,
                domains,
                types,
                modalities,
                benchmark_tasks,
                scores,
                gr.update(visible=show_zero_shot),
                initial_models,
            )

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
                return []
            benchmark_results = all_benchmark_results[benchmark_name]
            scores = benchmark_results._get_scores(languages=languages, format="long")
            elapsed = time.time() - start_time
            logger.debug(f"update_scores callback: {elapsed}s")
            return scores

        lang_select.input(
            update_scores_on_lang_change,
            inputs=[benchmark_select, lang_select],
            outputs=[scores],
        )

        @cachetools.cached(
            cache={},
            key=lambda benchmark_name,
            type_select,
            domain_select,
            lang_select,
            modality_select: hash(
                (
                    hash(benchmark_name),
                    hash(tuple(type_select)),
                    hash(tuple(domain_select)),
                    hash(tuple(lang_select)),
                    hash(tuple(modality_select)),
                )
            ),
        )
        def update_task_list(
            benchmark_name, type_select, domain_select, lang_select, modality_select
        ):
            if not len(lang_select):
                return []
            start_time = time.time()
            tasks_to_keep = []
            for task in mteb.get_benchmark(benchmark_name).tasks:
                if task.metadata.type not in type_select:
                    continue
                if task.metadata.domains is not None and not (
                    set(task.metadata.domains) & set(domain_select)
                ):
                    continue
                if task.languages is not None and not (
                    set(task.languages) & set(lang_select)
                ):
                    continue
                if task.metadata.modalities and not (
                    set(task.metadata.modalities) & set(modality_select)
                ):
                    continue
                tasks_to_keep.append(task.metadata.name)
            elapsed = time.time() - start_time
            logger.debug(f"update_task_list callback: {elapsed}s")
            return sorted(tasks_to_keep)

        type_select.input(
            update_task_list,
            inputs=[
                benchmark_select,
                type_select,
                domain_select,
                lang_select,
                modality_select,
            ],
            outputs=[task_select],
        )
        domain_select.input(
            update_task_list,
            inputs=[
                benchmark_select,
                type_select,
                domain_select,
                lang_select,
                modality_select,
            ],
            outputs=[task_select],
        )
        lang_select.input(
            update_task_list,
            inputs=[
                benchmark_select,
                type_select,
                domain_select,
                lang_select,
                modality_select,
            ],
            outputs=[task_select],
        )
        modality_select.input(
            update_task_list,
            inputs=[
                benchmark_select,
                type_select,
                domain_select,
                lang_select,
                modality_select,
            ],
            outputs=[task_select],
        )

        @cachetools.cached(
            cache={},
            key=lambda scores,
            tasks,
            availability,
            compatibility,
            instructions,
            max_model_size,
            zero_shot: hash(
                (
                    id(scores),
                    hash(tuple(tasks)),
                    hash(availability),
                    hash(tuple(compatibility)),
                    hash(instructions),
                    hash(max_model_size),
                    hash(zero_shot),
                )
            ),
        )
        def update_models(
            scores: list[dict],
            tasks: list[str],
            availability: bool | None,
            compatibility: list[str],
            instructions: bool | None,
            max_model_size: int,
            zero_shot: Literal["allow_all", "remove_unknown", "only_zero_shot"],
        ):
            start_time = time.time()
            model_names = list({entry["model_name"] for entry in scores})
            filtered_models = _filter_models(
                model_names,
                tasks,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot_setting=zero_shot,
            )
            elapsed = time.time() - start_time
            logger.debug(f"update_models callback: {elapsed}s")
            # Always return sorted models to ensure models.change triggers update_tables

            return sorted(filtered_models)

        scores.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )

        task_select.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        availability.input(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        compatibility.input(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        instructions.input(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        max_model_size.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        zero_shot.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                max_model_size,
                zero_shot,
            ],
            outputs=[models],
        )

        def _cache_key_for_update_tables(scores, tasks, models_to_keep, benchmark_name):
            scores_hash = hash(
                tuple(sorted((d.get("model_name"), d.get("revision")) for d in scores))
            )
            tasks_hash = hash(tuple(sorted(tasks)))
            # Sort models_to_keep to ensure consistent hash regardless of input order
            models_hash = (
                hash(tuple(sorted(models_to_keep)))
                if models_to_keep is not None
                else None
            )
            bench_hash = hash(benchmark_name)
            key = hash((scores_hash, tasks_hash, models_hash, bench_hash))

            return key

        @cachetools.cached(
            cache={},
            key=_cache_key_for_update_tables,
        )
        def update_tables(
            scores,
            tasks,
            models_to_keep,
            benchmark_name: str,
        ):
            start_time = time.time()
            tasks = set(tasks)
            benchmark = mteb.get_benchmark(benchmark_name)
            benchmark_tasks = {task.metadata.name for task in benchmark.tasks}

            # Extract filtered model and task names from scores (respects UI filters)
            filtered_model_names = set()
            filtered_task_names = set()

            for entry in scores:
                if entry["task_name"] not in tasks:
                    continue
                if (models_to_keep is not None) and (
                    entry["model_name"] not in models_to_keep
                ):
                    continue
                filtered_model_names.add(entry["model_name"])
                filtered_task_names.add(entry["task_name"])

            # Create filtered BenchmarkResults as required by Kenneth
            benchmark_results = all_benchmark_results[benchmark_name]
            filtered_benchmark_results = benchmark_results

            # Apply task filtering if needed
            if filtered_task_names != benchmark_tasks:
                filtered_benchmark_results = filtered_benchmark_results._filter_tasks(
                    task_names=list(filtered_task_names)
                )

            # Apply model filtering if needed
            if filtered_model_names:
                filtered_benchmark_results = filtered_benchmark_results.select_models(
                    list(filtered_model_names)
                )

            summary = apply_summary_styling_from_benchmark(
                benchmark, filtered_benchmark_results
            )
            per_task = apply_per_task_styling_from_benchmark(
                benchmark, filtered_benchmark_results
            )
            elapsed = time.time() - start_time
            logger.debug(f"update_tables callback: {elapsed}s")
            return summary, per_task

        # Only update tables when models change, not when scores/tasks change directly
        # This avoids redundant updates since scores/tasks changes trigger update_models
        # which then triggers models.change
        for item in [models, task_select]:
            item.change(
                update_tables,
                inputs=[scores, task_select, models, benchmark_select],
                outputs=[summary_table, per_task_table],
            )

        gr.Markdown(ACKNOWLEDGEMENT, elem_id="ack_markdown")

    # Prerun on all benchmarks, so that results of callbacks get cached
    for benchmark in benchmarks:
        (
            bench_languages,
            bench_domains,
            bench_types,
            bench_modalities,
            bench_tasks,
            bench_scores,
            zero_shot,
            bench_initial_models,
        ) = on_benchmark_select(benchmark.name)
        # Call update_tables to populate cache (simulating models.change trigger)
        update_tables(bench_scores, bench_tasks, bench_initial_models, benchmark.name)
        # Also cache the filtered tasks scenario
        filtered_tasks = update_task_list(
            benchmark.name,
            bench_types,
            bench_domains,
            bench_languages,
            bench_modalities,
        )
        update_tables(
            bench_scores, filtered_tasks, bench_initial_models, benchmark.name
        )
    return demo


if __name__ == "__main__":
    logging.getLogger("mteb.load_results.task_results").setLevel(
        logging.ERROR
    )  # Warnings related to task split
    logging.getLogger("mteb.model_meta").setLevel(
        logging.ERROR
    )  # Warning related to model metadata (fetch_from_hf=False)
    logging.getLogger("mteb.load_results.benchmark_results").setLevel(
        logging.ERROR
    )  # Warning related to model metadata (fetch_from_hf=False)
    warnings.filterwarnings("ignore", message="Couldn't get scores for .* due to .*")

    app = get_leaderboard_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
