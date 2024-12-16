from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode

import gradio as gr
import pandas as pd
from gradio_rangeslider import RangeSlider

import mteb
from mteb.caching import json_cache
from mteb.leaderboard.figures import performance_size_plot, radar_chart
from mteb.leaderboard.table import scores_to_tables


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = mteb.load_results()
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


def set_benchmark_on_load(request: gr.Request):
    query_params = request.query_params
    return query_params.get("benchmark_name", "MTEB(Multilingual, beta)")


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


all_results = load_results().join_revisions().filter_models()

# Model sizes in million parameters
min_model_size, max_model_size = 0, 10_000

benchmarks = mteb.get_benchmarks()

default_benchmark = mteb.get_benchmark("MTEB(Multilingual, beta)")
default_results = default_benchmark.load_results(base_results=all_results)

default_scores = default_results.get_scores(format="long")
summary_table, per_task_table = scores_to_tables(default_scores)

benchmark_select = gr.Dropdown(
    [bench.name for bench in benchmarks],
    value=default_benchmark.name,
    label="Prebuilt Benchmarks",
    info="Select one of our expert-selected benchmarks from MTEB publications.",
)
lang_select = gr.Dropdown(
    all_results.languages,
    value=default_results.languages,
    multiselect=True,
    label="Language",
    info="Select languages to include.",
)
type_select = gr.Dropdown(
    all_results.task_types,
    value=default_results.task_types,
    multiselect=True,
    label="Task Type",
    info="Select task types to include.",
)
domain_select = gr.Dropdown(
    all_results.domains,
    value=default_results.domains,
    multiselect=True,
    label="Domain",
    info="Select domains to include.",
)
task_select = gr.Dropdown(
    all_results.task_names,
    value=default_results.task_names,
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
                searchbar = gr.Textbox(
                    label="Search Models",
                    info="Search models by name (RegEx sensitive. Separate queries with `|`)",
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
                        model_size = RangeSlider(
                            minimum=min_model_size,
                            maximum=max_model_size,
                            value=(min_model_size, max_model_size),
                            label="Model Size (#M Parameters)",
                            interactive=True,
                        )
    scores = gr.State(default_scores)
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
        summary_table.render()
        download_summary = gr.DownloadButton("Download Table")
        download_summary.click(
            download_table, inputs=[summary_table], outputs=[download_summary]
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

    @gr.on(inputs=[scores, searchbar], outputs=[summary_table, per_task_table])
    def update_tables(scores, search_query: str):
        summary, per_task = scores_to_tables(scores, search_query)
        return summary, per_task

    @gr.on(
        inputs=[benchmark_select],
        outputs=[
            lang_select,
            type_select,
            domain_select,
        ],
    )
    @json_cache
    def on_select_benchmark(benchmark_name):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        task_types = benchmark_results.task_types
        langs = benchmark_results.languages
        domains = benchmark_results.domains
        return (
            langs,
            task_types,
            domains,
        )

    @gr.on(
        inputs=[benchmark_select, lang_select, type_select, domain_select],
        outputs=[task_select],
    )
    @json_cache
    def update_task_list(benchmark_name, languages, task_types, domains):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        task_to_lang_set = defaultdict(set)
        task_to_type = {}
        task_to_domains = defaultdict(set)
        for model_res in benchmark_results:
            for task_res in model_res:
                task_to_lang_set[task_res.task_name] |= set(task_res.languages)
                task_to_domains[task_res.task_name] |= set(task_res.domains)
                task_to_type[task_res.task_name] = task_res.task_type
        res = []
        for task_name in benchmark_results.task_names:
            if not (task_to_domains[task_name] & set(domains)):
                continue
            if not (task_to_lang_set[task_name] & set(languages)):
                continue
            if task_to_type[task_name] not in task_types:
                continue
            res.append(task_name)
        return res

    @gr.on(
        inputs=[
            benchmark_select,
            task_select,
            lang_select,
            type_select,
            domain_select,
            availability,
            compatibility,
            instructions,
            model_size,
        ],
        outputs=[scores],
    )
    def update_scores(
        benchmark_name,
        task_names,
        languages,
        task_types,
        domains,
        availability,
        compatibility,
        instructions,
        model_size,
    ):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        benchmark_results = benchmark_results.filter_tasks(
            languages=languages,
            task_names=task_names,
            task_types=task_types,
            domains=domains,
        )
        lower, upper = model_size
        # Setting to None, when the user doesn't specify anything
        if (lower == min_model_size) and (upper == max_model_size):
            lower, upper = None, None
        else:
            # Multiplying by millions
            lower = lower * 1e6
            upper = upper * 1e6
        benchmark_results = benchmark_results.filter_models(
            open_weights=availability,
            use_instructions=instructions,
            frameworks=compatibility,
            n_parameters_range=(lower, upper),
        )
        scores = benchmark_results.get_scores(languages=languages, format="long")
        return scores


if __name__ == "__main__":
    demo.launch()
