import functools
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from gradio_rangeslider import RangeSlider

import mteb
from mteb.leaderboard.table import scores_to_table
from mteb.leaderboard.utils import get_model_size_range


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = mteb.load_results()
        all_results.to_disk(results_cache_path)
        return all_results
    else:
        return mteb.BenchmarkResults.from_disk(results_cache_path)


all_results = load_results().filter_models()

max_model_size, min_model_size = get_model_size_range()

benchmarks = mteb.get_benchmarks()

default_benchmark = mteb.get_benchmark("MTEB(multilingual)")
default_results = default_benchmark.load_results(base_results=all_results)

benchmark_select = gr.Dropdown(
    [bench.name for bench in benchmarks],
    value=default_benchmark.name,
    label="Prebuilt Benchmarks",
    info="Select one of our expert-selected benchmarks from MTEB publications.",
)
lang_select = gr.Dropdown(
    default_results.languages,
    value=default_results.languages,
    multiselect=True,
    label="Language",
    info="Select languages to include.",
)
type_select = gr.Dropdown(
    default_results.task_types,
    value=default_results.task_types,
    multiselect=True,
    label="Task Type",
    info="Select task types to include.",
)
domain_select = gr.Dropdown(
    default_results.domains,
    value=default_results.domains,
    multiselect=True,
    label="Domain",
    info="Select domains to include.",
)
task_select = gr.Dropdown(
    default_results.task_names,
    value=default_results.task_names,
    multiselect=True,
    label="Task",
    info="Select specific tasks to include",
)

css = """
.scrollable {
    overflow-y: scroll;
    max-height: 400px
}
"""

with gr.Blocks(fill_width=True, theme=gr.themes.Base(), css=css) as demo:
    gr.Markdown(
        """
    ### Model Selection
    Select models to rank based on an assortment of criteria. 
    """
    )
    with gr.Group():
        with gr.Row():
            with gr.Column():
                availability = gr.Radio(
                    [("Only Open", True), ("Only Proprietary", False), ("Both", None)],
                    value=None,
                    label="Availability",
                    interactive=True,
                )
                compatibility = gr.CheckboxGroup(
                    [
                        (
                            "Should be sentence-transformers compatible",
                            "sbert_compatible",
                        )
                    ],
                    value=[],
                    label="Compatibility",
                    interactive=True,
                )
            with gr.Column():
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
                model_size = RangeSlider(
                    minimum=0,
                    maximum=8000,
                    value=(0, 8000),
                    label="Model Size (#M Parameters)",
                    interactive=True,
                )

    gr.Markdown(
        """
    ### Benchmarks
    Select one of the hand-curated benchmarks from our publication.
    Or create one from scratch based on your use case.
    """
    )
    with gr.Group(elem_classes="scrollable"):
        with gr.Row():
            with gr.Column():
                benchmark_select.render()
                with gr.Row():
                    lang_select.render()
                    type_select.render()
                with gr.Row():
                    domain_select.render()
            with gr.Column():
                # with gr.Accordion("Add and remove tasks:", open=False):
                task_select.render()
    scores = gr.State(default_results.get_scores(format="long"))
    dataframe = gr.DataFrame(
        scores_to_table,
        inputs=[scores],
    )

    @gr.on(
        inputs=[benchmark_select],
        outputs=[
            lang_select,
            type_select,
            domain_select,
        ],
    )
    def on_select_benchmark(benchmark_name):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        return (
            benchmark_results.languages,
            benchmark_results.task_types,
            benchmark_results.domains,
        )

    @gr.on(
        inputs=[benchmark_select, lang_select, type_select, domain_select],
        outputs=[task_select],
    )
    def update_task_list(benchmark_name, languages, task_types, domains):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        return benchmark_results.task_names

    @gr.on(
        inputs=[
            benchmark_select,
            task_select,
            lang_select,
            type_select,
            domain_select,
        ],
        outputs=[scores],
    )
    def update_scores(benchmark_name, task_names, languages, task_types, domains):
        benchmark = mteb.get_benchmark(benchmark_name)
        benchmark_results = benchmark.load_results(base_results=all_results)
        benchmark_results = benchmark_results.filter_tasks(
            languages=languages,
            task_names=task_names,
            task_types=task_types,
            domains=domains,
        )
        scores = benchmark_results.get_scores(languages=languages, format="long")
        return scores


if __name__ == "__main__":
    demo.launch()
