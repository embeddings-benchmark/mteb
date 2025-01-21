from __future__ import annotations

import itertools
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import gradio as gr
import pandas as pd
from gradio_rangeslider import RangeSlider

import mteb
from mteb.caching import json_cache
from mteb.leaderboard.figures import performance_size_plot, radar_chart
from mteb.leaderboard.table import scores_to_tables

logger = logging.getLogger(__name__)


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = (
            mteb.load_results(only_main_score=True, require_model_meta=False)
            .join_revisions()
            .filter_models()
        )
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


DEFAULT_BENCHMARK_NAME = "MTEB(Multilingual)"


def set_benchmark_on_load(request: gr.Request):
    query_params = request.query_params
    return query_params.get("benchmark_name", DEFAULT_BENCHMARK_NAME)


def download_table(table: pd.DataFrame) -> Path:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    table.to_csv(file)
    return file.name


def update_citation(benchmark_name: str) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    if str(benchmark.citation) != "None":
        citation = f"```bibtex\n{benchmark.citation}\n```"
    else:
        citation = ""
    return citation


def update_description(
    benchmark_name: str, languages: list[str], task_types: list[str], domains: list[str]
) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    description = f"## {benchmark.name}\n{benchmark.description}\n"
    n_languages = len(languages)
    n_task_types = len(task_types)
    n_tasks = len(benchmark.tasks)
    n_domains = len(domains)
    description += f" - **Number of languages**: {n_languages}\n"
    description += f" - **Number of datasets**: {n_tasks}\n"
    description += f" - **Number of task types**: {n_task_types}\n"
    description += f" - **Number of domains**: {n_domains}\n"
    if str(benchmark.reference) != "None":
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
        properties=["name", "type", "languages", "domains", "reference", "main_score"]
    )
    df["languages"] = df["languages"].map(format_list)
    df = df.sort_values("name")
    df["domains"] = df["domains"].map(format_list)
    df["name"] = "[" + df["name"] + "](" + df["reference"] + ")"
    df = df.rename(
        columns={
            "name": "Task Name",
            "type": "Task Type",
            "languages": "Languages",
            "domains": "Domains",
            "main_score": "Metric",
        }
    )
    df = df.drop(columns="reference")
    return gr.DataFrame(df, datatype=["markdown"] + ["str"] * (len(df.columns) - 1))


# Model sizes in million parameters
MIN_MODEL_SIZE, MAX_MODEL_SIZE = 0, 10_000


def filter_models(
    model_names,
    task_select,
    availability,
    compatibility,
    instructions,
    model_size,
    zero_shot_setting,
):
    lower, upper = model_size
    # Setting to None, when the user doesn't specify anything
    if (lower == MIN_MODEL_SIZE) and (upper == MAX_MODEL_SIZE):
        lower, upper = None, None
    else:
        # Multiplying by millions
        lower = lower * 1e6
        upper = upper * 1e6
    model_metas = mteb.get_model_metas(
        model_names=model_names,
        open_weights=availability,
        use_instructions=instructions,
        frameworks=compatibility,
        n_parameters_range=(lower, upper),
    )
    tasks = mteb.get_tasks(tasks=task_select)
    models_to_keep = set()
    for model_meta in model_metas:
        is_model_zero_shot = model_meta.is_zero_shot_on(tasks)
        if is_model_zero_shot is None:
            if zero_shot_setting == "hard":
                continue
        elif not is_model_zero_shot:
            if zero_shot_setting != "off":
                continue
        models_to_keep.add(model_meta.name)
    return list(models_to_keep)


logger.info("Loading all benchmark results")
all_results = load_results()

benchmarks = mteb.get_benchmarks()
all_benchmark_results = {
    benchmark.name: benchmark.load_results(base_results=all_results)
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
    zero_shot_setting="soft",
)

summary_table, per_task_table = scores_to_tables(
    [entry for entry in default_scores if entry["model_name"] in filtered_models]
)

benchmark_select = gr.Dropdown(
    [bench.name for bench in benchmarks],
    value=default_benchmark.name,
    label="Prebuilt Benchmarks",
    info="Select one of our expert-selected benchmarks from MTEB publications.",
)
lang_select = gr.Dropdown(
    all_results.languages,
    value=sorted(default_results.languages),
    multiselect=True,
    label="Language",
    info="Select languages to include.",
)
type_select = gr.Dropdown(
    all_results.task_types,
    value=sorted(default_results.task_types),
    multiselect=True,
    label="Task Type",
    info="Select task types to include.",
)
domain_select = gr.Dropdown(
    all_results.domains,
    value=sorted(default_results.domains),
    multiselect=True,
    label="Domain",
    info="Select domains to include.",
)
task_select = gr.Dropdown(
    all_results.task_names,
    value=sorted(default_results.task_names),
    allow_custom_value=True,
    multiselect=True,
    label="Task",
    info="Select specific tasks to include",
)

head = """
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
"""

with gr.Blocks(fill_width=True, theme=gr.themes.Base(), head=head) as demo:
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown(
                """
            ### Benchmarks
            Select one of the hand-curated benchmarks from our publications and modify them using one of the following filters to fit your needs.
            """
            )
            with gr.Group():
                with gr.Row(elem_classes="overflow-y-scroll max-h-80"):
                    with gr.Column():
                        benchmark_select.render()
                        with gr.Accordion("Select Languages", open=False):
                            lang_select.render()
                        with gr.Accordion("Select Task Types", open=False):
                            type_select.render()
                        with gr.Accordion("Select Domains", open=False):
                            domain_select.render()
                        with gr.Accordion("Add and remove tasks:", open=False):
                            task_select.render()
        with gr.Column(scale=8):
            gr.Markdown(
                """
            ### Model Selection
            Select models to rank based on an assortment of criteria. 
            """,
            )
            with gr.Group():
                with gr.Row():
                    searchbar = gr.Textbox(
                        label="Search Models",
                        info="Search models by name (RegEx sensitive. Separate queries with `|`)",
                        interactive=True,
                    )
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
                with gr.Row(elem_classes=""):
                    with gr.Column():
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
                                    "hard",
                                ),
                                ("Allow Unknown", "soft"),
                                ("Allow all", "off"),
                            ],
                            value="soft",
                            label="Zero-shot",
                            interactive=True,
                        )
                        model_size = RangeSlider(
                            minimum=MIN_MODEL_SIZE,
                            maximum=MAX_MODEL_SIZE,
                            value=(MIN_MODEL_SIZE, MAX_MODEL_SIZE),
                            label="Model Size (#M Parameters)",
                            interactive=True,
                        )
    scores = gr.State(default_scores)
    models = gr.State(filtered_models)
    with gr.Row():
        with gr.Column():
            description = gr.Markdown(
                update_description,
                inputs=[benchmark_select, lang_select, type_select, domain_select],
            )
            citation = gr.Markdown(update_citation, inputs=[benchmark_select])
            with gr.Accordion("Share this benchmark:", open=False):
                gr.Markdown(produce_benchmark_link, inputs=[benchmark_select])
        with gr.Column():
            with gr.Tab("Performance per Model Size"):
                plot = gr.Plot(performance_size_plot, inputs=[summary_table])
                gr.Markdown(
                    "*We only display models that have been run on all tasks in the benchmark*"
                )
            with gr.Tab("Performance per Task Type (Radar Chart)"):
                radar_plot = gr.Plot(radar_chart, inputs=[summary_table])
                gr.Markdown(
                    "*We only display models that have been run on all task types in the benchmark*"
                )
    with gr.Tab("Summary"):
        gr.Markdown(
            """
            ✅ - Model is zero-shot on the benchmark <br>
            ⚠️  - Training data unknown <br>
            ❌ - Model is **NOT** zero-shot on the benchmark
        """
        )
        summary_table.render()
        download_summary = gr.DownloadButton("Download Table")
        download_summary.click(
            download_table, inputs=[summary_table], outputs=[download_summary]
        )
        with gr.Accordion(
            "What do aggregate measures (Rank(Borda), Mean(Task), etc.) mean?",
            open=False,
        ):
            gr.Markdown(
                """
    **Rank(borda)** is computed based on the [borda count](https://en.wikipedia.org/wiki/Borda_count), where each task is treated as a preference voter, which gives votes on the models in accordance with their relative performance on the task. The best model obtains the highest number of votes. The model with the highest number of votes across tasks obtains the highest rank. The Borda rank tends to prefer models that perform well broadly across tasks. However, given that it is a rank it can be unclear if the two models perform similarly.

    **Mean(Task)**: This is a naïve average computed across all the tasks within the benchmark. This score is simple to understand and is continuous as opposed to the Borda rank. However, the mean can overvalue tasks with higher variance in its scores. 

    **Mean(TaskType)**: This is a weighted average across different task categories, such as classification or retrieval. It is computed by first computing the average by task category and then computing the average on each category. Similar to the Mean(Task) this measure is continuous and tends to overvalue tasks with higher variance. This score also prefers models that perform well across all task categories.
            """
            )
        with gr.Accordion(
            "What does zero-shot mean?",
            open=False,
        ):
            gr.Markdown(
                """
A model is considered zero-shot if it is not trained on any splits of the datasets used to derive the tasks.
E.g., if a model is trained on Natural Questions, it cannot be considered zero-shot on benchmarks containing the task “NQ” which is derived from Natural Questions.
This definition creates a few edge cases. For instance, multiple models are typically trained on Wikipedia title and body pairs, but we do not define this as leakage on, e.g., “WikipediaRetrievalMultilingual” and “WikiClusteringP2P” as these datasets are not based on title-body pairs.
Distilled, further fine-tunes or in other ways, derivative models inherit the datasets of their parent models.
Based on community feedback and research findings, This definition could change in the future.
            """
            )
    with gr.Tab("Performance per task"):
        per_task_table.render()
        download_per_task = gr.DownloadButton("Download Table")
        download_per_task.click(
            download_table, inputs=[per_task_table], outputs=[download_per_task]
        )
    with gr.Tab("Task information"):
        task_info_table = gr.DataFrame(update_task_info, inputs=[task_select])

    # This sets the benchmark from the URL query parameters
    demo.load(set_benchmark_on_load, inputs=[], outputs=[benchmark_select])

    @json_cache
    def on_benchmark_select(benchmark_name):
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
        languages, domains, types = (
            sorted(languages),
            sorted(domains),
            sorted(types),
        )
        elapsed = time.time() - start_time
        benchmark_results = all_benchmark_results[benchmark_name]
        scores = benchmark_results.get_scores(format="long")
        logger.info(f"on_benchmark_select callback: {elapsed}s")
        return (
            languages,
            domains,
            types,
            [task.metadata.name for task in benchmark.tasks],
            scores,
        )

    benchmark_select.change(
        on_benchmark_select,
        inputs=[benchmark_select],
        outputs=[lang_select, domain_select, type_select, task_select, scores],
    )

    @json_cache
    def update_scores_on_lang_change(benchmark_name, languages):
        start_time = time.time()
        benchmark_results = all_benchmark_results[benchmark_name]
        scores = benchmark_results.get_scores(languages=languages, format="long")
        elapsed = time.time() - start_time
        logger.info(f"update_scores callback: {elapsed}s")
        return scores

    lang_select.input(
        update_scores_on_lang_change,
        inputs=[benchmark_select, lang_select],
        outputs=[scores],
    )

    def update_task_list(benchmark_name, type_select, domain_select, lang_select):
        start_time = time.time()
        tasks_to_keep = []
        for task in mteb.get_benchmark(benchmark_name).tasks:
            if task.metadata.type not in type_select:
                continue
            if not (set(task.metadata.domains or []) & set(domain_select)):
                continue
            if not (set(task.languages or []) & set(lang_select)):
                continue
            tasks_to_keep.append(task.metadata.name)
        elapsed = time.time() - start_time
        logger.info(f"update_task_list callback: {elapsed}s")
        return tasks_to_keep

    type_select.input(
        update_task_list,
        inputs=[benchmark_select, type_select, domain_select, lang_select],
        outputs=[task_select],
    )
    domain_select.input(
        update_task_list,
        inputs=[benchmark_select, type_select, domain_select, lang_select],
        outputs=[task_select],
    )
    lang_select.input(
        update_task_list,
        inputs=[benchmark_select, type_select, domain_select, lang_select],
        outputs=[task_select],
    )

    def update_models(
        scores: list[dict],
        tasks: list[str],
        availability: bool | None,
        compatibility: list[str],
        instructions: bool | None,
        model_size: tuple[int, int],
        zero_shot: Literal["hard", "soft", "off"],
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
        logger.info(f"update_models callback: {elapsed}s")
        return filtered_models

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
    model_size.input(
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

    def update_tables(
        scores,
        search_query: str,
        tasks,
        models_to_keep,
    ):
        start_time = time.time()
        tasks = set(tasks)
        models_to_keep = set(models_to_keep)
        filtered_scores = []
        for entry in scores:
            if entry["task_name"] not in tasks:
                continue
            if entry["model_name"] not in models_to_keep:
                continue
            filtered_scores.append(entry)
        summary, per_task = scores_to_tables(filtered_scores, search_query)
        elapsed = time.time() - start_time
        logger.info(f"update_tables callback: {elapsed}s")
        return summary, per_task

    task_select.change(
        update_tables,
        inputs=[scores, searchbar, task_select, models],
        outputs=[summary_table, per_task_table],
    )
    scores.change(
        update_tables,
        inputs=[scores, searchbar, task_select, models],
        outputs=[summary_table, per_task_table],
    )
    models.change(
        update_tables,
        inputs=[scores, searchbar, task_select, models],
        outputs=[summary_table, per_task_table],
    )
    searchbar.input(
        update_tables,
        inputs=[scores, searchbar, task_select, models],
        outputs=[summary_table, per_task_table],
    )


if __name__ == "__main__":
    demo.launch()
