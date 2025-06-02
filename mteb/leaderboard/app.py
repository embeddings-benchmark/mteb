from __future__ import annotations

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
from gradio_rangeslider import RangeSlider

import mteb
from mteb.abstasks.TaskMetadata import TASK_DOMAIN, TASK_TYPE
from mteb.benchmarks.benchmarks import MTEB_multilingual
from mteb.custom_validators import MODALITIES
from mteb.leaderboard.benchmark_selector import BENCHMARK_ENTRIES, make_selector
from mteb.leaderboard.figures import performance_size_plot, radar_chart
from mteb.leaderboard.table import create_tables
from mteb.leaderboard.text_segments import ACKNOWLEDGEMENT, FAQ

logger = logging.getLogger(__name__)


LANGUAGE: list[str] = list({l for t in mteb.get_tasks() for l in t.metadata.languages})
ALL_MODELS = {meta.name for meta in mteb.get_model_metas()}


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = mteb.load_results(
            only_main_score=True, require_model_meta=False, models=ALL_MODELS
        ).filter_models()
        all_results.to_disk(results_cache_path)
        return all_results
    else:
        with results_cache_path.open() as cache_file:
            return mteb.BenchmarkResults.from_validated(**json.load(cache_file))


def produce_benchmark_link(benchmark_name: str, request: gr.Request) -> str:
    """Produces a URL for the selected benchmark."""
    params = urlencode(
        {
            "benchmark_name": benchmark_name,
        }
    )
    base_url = request.request.base_url
    url = f"{base_url}?{params}"
    md = f"```\n{url}\n```"
    return md


DEFAULT_BENCHMARK_NAME = MTEB_multilingual.name


def set_benchmark_on_load(request: gr.Request):
    query_params = request.query_params
    return query_params.get("benchmark_name", DEFAULT_BENCHMARK_NAME)


def download_table(table: pd.DataFrame) -> str:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    table.to_csv(file)
    return file.name


def update_citation(benchmark_name: str) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    if benchmark.citation is not None:
        citation = f"```bibtex\n{benchmark.citation}\n```"
    else:
        citation = ""
    return citation


def update_description(
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
        description += f"\n[Click for More Info]({benchmark.reference})"

    return description


def format_list(props: list[str]):
    if props is None:
        return ""
    if len(props) > 3:
        return ", ".join(props[:3]) + "..."
    return ", ".join(props)


def update_task_info(task_names: str) -> gr.DataFrame:
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
        ]
    )
    df["languages"] = df["languages"].map(format_list)
    df = df.sort_values("name")
    df["domains"] = df["domains"].map(format_list)
    df["name"] = "[" + df["name"] + "](" + df["reference"] + ")"
    df["modalities"] = df["modalities"].map(format_list)
    df = df.rename(
        columns={
            "name": "Task Name",
            "type": "Task Type",
            "languages": "Languages",
            "domains": "Domains",
            "main_score": "Metric",
            "modalities": "Modality",
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


def filter_models(
    model_names: list[str],
    task_select: list[str],
    availability: bool | None,
    compatibility: list[str],
    instructions: bool | None,
    model_size: tuple[int | None, int | None],
    zero_shot_setting: Literal["only_zero_shot", "allow_all", "remove_unknown"],
):
    lower, upper = model_size
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


def get_leaderboard_app() -> gr.Blocks:
    logger.info("Loading all benchmark results")
    all_results = load_results()

    benchmarks = sorted(
        mteb.get_benchmarks(display_on_leaderboard=True), key=lambda x: x.name
    )
    all_benchmark_results = {
        benchmark.name: benchmark.load_results(
            base_results=all_results
        ).join_revisions()
        for benchmark in benchmarks
    }
    default_benchmark = mteb.get_benchmark(DEFAULT_BENCHMARK_NAME)
    default_results = all_benchmark_results[default_benchmark.name]
    logger.info("Benchmark results loaded")

    default_scores = default_results.get_scores(format="long")
    all_models = list({entry["model_name"] for entry in default_scores})
    filtered_models = filter_models(
        all_models,
        default_results.task_names,
        availability=None,
        compatibility=[],
        instructions=None,
        model_size=(MIN_MODEL_SIZE, MAX_MODEL_SIZE),
        zero_shot_setting="allow_all",
    )

    summary_table, per_task_table = create_tables(
        [entry for entry in default_scores if entry["model_name"] in filtered_models]
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
        sorted(get_args(TASK_TYPE)),
        value=sorted(default_results.task_types),
        multiselect=True,
        label="Task Type",
        info="Select task types to include.",
    )
    domain_select = gr.Dropdown(
        sorted(get_args(TASK_DOMAIN)),
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
        sorted(get_args(MODALITIES)),
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
            gr.Markdown("## Select Benchmark")
            benchmark_select, column = make_selector(BENCHMARK_ENTRIES)
        gr.Markdown(
            """
        ## Embedding Leaderboard

        This leaderboard compares 100+ text and image embedding models across 1000+ languages. We refer to the publication of each selectable benchmark for details on metrics, languages, tasks, and task types. Anyone is welcome [to add a model](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md), [add benchmarks](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_benchmark.md), [help us improve zero-shot annotations](https://github.com/embeddings-benchmark/mteb/blob/06489abca007261c7e6b11f36d4844c5ed5efdcb/mteb/models/bge_models.py#L91) or [propose other changes to the leaderboard](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/leaderboard).
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
                    update_description,
                    inputs=[benchmark_select, lang_select, type_select, domain_select],
                )
                with gr.Accordion("Cite this benchmark:", open=False):
                    citation = gr.Markdown(update_citation, inputs=[benchmark_select])  # noqa: F841
                with gr.Accordion("Share this benchmark:", open=False):
                    gr.Markdown(produce_benchmark_link, inputs=[benchmark_select])
            with gr.Column(scale=2):
                with gr.Tab("Performance per Model Size"):
                    plot = gr.Plot(performance_size_plot, inputs=[summary_table])  # noqa: F841
                    gr.Markdown(
                        "*We only display models that have been run on all tasks in the benchmark*"
                    )
                with gr.Tab("Performance per Task Type (Radar Chart)"):
                    radar_plot = gr.Plot(radar_chart, inputs=[summary_table])  # noqa: F841
                    gr.Markdown(
                        "*We only display models that have been run on all task types in the benchmark*"
                    )

        with gr.Accordion("Customize this Benchmark", open=False):
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
                        model_size = RangeSlider(
                            minimum=MIN_MODEL_SIZE,
                            maximum=MAX_MODEL_SIZE,
                            value=(MIN_MODEL_SIZE, MAX_MODEL_SIZE),
                            label="Model Size (#M Parameters)",
                        )

        with gr.Tab("Summary"):
            summary_table.render()
            download_summary = gr.DownloadButton("Download Table")
            download_summary.click(
                download_table, inputs=[summary_table], outputs=[download_summary]
            )

            with gr.Accordion(
                "Frequently Asked Questions",
                open=False,
            ):
                gr.Markdown(FAQ)
        with gr.Tab("Performance per task"):
            per_task_table.render()
            download_per_task = gr.DownloadButton("Download Table")
            download_per_task.click(
                download_table, inputs=[per_task_table], outputs=[download_per_task]
            )
        with gr.Tab("Task information"):
            task_info_table = gr.DataFrame(update_task_info, inputs=[task_select])  # noqa: F841

        # This sets the benchmark from the URL query parameters
        demo.load(set_benchmark_on_load, inputs=[], outputs=[benchmark_select])

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
            scores = benchmark_results.get_scores(format="long")
            logger.debug(f"on_benchmark_select callback: {elapsed}s")
            return (
                languages,
                domains,
                types,
                modalities,
                sorted([task.metadata.name for task in benchmark.tasks]),
                scores,
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
            scores = benchmark_results.get_scores(languages=languages, format="long")
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
            model_size,
            zero_shot: hash(
                (
                    id(scores),
                    hash(tuple(tasks)),
                    hash(availability),
                    hash(tuple(compatibility)),
                    hash(instructions),
                    hash(model_size),
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
            model_size: tuple[int, int],
            zero_shot: Literal["allow_all", "remove_unknown", "only_zero_shot"],
        ):
            start_time = time.time()
            model_names = list({entry["model_name"] for entry in scores})
            filtered_models = filter_models(
                model_names,
                tasks,
                availability,
                compatibility,
                instructions,
                model_size,
                zero_shot_setting=zero_shot,
            )
            elapsed = time.time() - start_time
            if model_names == filtered_models:
                # This indicates that the models should not be filtered
                return None
            logger.debug(f"update_models callback: {elapsed}s")
            return sorted(filtered_models)

        scores.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                model_size,
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
                model_size,
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
                model_size,
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
                model_size,
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
                model_size,
                zero_shot,
            ],
            outputs=[models],
        )
        model_size.change(
            update_models,
            inputs=[
                scores,
                task_select,
                availability,
                compatibility,
                instructions,
                model_size,
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
                model_size,
                zero_shot,
            ],
            outputs=[models],
        )

        @cachetools.cached(
            cache={},
            key=lambda scores, tasks, models_to_keep, benchmark_name: hash(
                (
                    id(scores),
                    hash(tuple(tasks)),
                    id(models_to_keep),
                    hash(benchmark_name),
                )
            ),
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
            if (benchmark_tasks != tasks) or (models_to_keep is not None):
                filtered_scores = []
                for entry in scores:
                    if entry["task_name"] not in tasks:
                        continue
                    if (models_to_keep is not None) and (
                        entry["model_name"] not in models_to_keep
                    ):
                        continue
                    filtered_scores.append(entry)
            else:
                filtered_scores = scores
            summary, per_task = create_tables(filtered_scores)
            elapsed = time.time() - start_time
            logger.debug(f"update_tables callback: {elapsed}s")
            return summary, per_task

        task_select.change(
            update_tables,
            inputs=[scores, task_select, models, benchmark_select],
            outputs=[summary_table, per_task_table],
        )
        scores.change(
            update_tables,
            inputs=[scores, task_select, models, benchmark_select],
            outputs=[summary_table, per_task_table],
        )
        models.change(
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
        ) = on_benchmark_select(benchmark.name)
        filtered_models = update_models(
            bench_scores,
            bench_tasks,
            availability=None,
            compatibility=[],
            instructions=None,
            model_size=(MIN_MODEL_SIZE, MAX_MODEL_SIZE),
            zero_shot="allow_all",
        )
        # We have to call this both on the filtered and unfiltered task because the callbacks
        # also gets called twice for some reason
        update_tables(bench_scores, bench_tasks, filtered_models, benchmark.name)
        filtered_tasks = update_task_list(
            benchmark.name,
            bench_types,
            bench_domains,
            bench_languages,
            bench_modalities,
        )
        update_tables(bench_scores, filtered_tasks, filtered_models, benchmark.name)
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
