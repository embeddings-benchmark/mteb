import functools
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from gradio_rangeslider import RangeSlider

import mteb
from mteb.benchmarks.benchmarks import Benchmark
from mteb.leaderboard.utils import (get_domains, get_languages,
                                    get_model_size_range, get_task_types)


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_tasks = mteb.get_tasks()
        all_results = mteb.load_results(tasks=all_tasks)
        all_results.to_disk(results_cache_path)
        return all_results
    else:
        return mteb.BenchmarkResults.from_disk(results_cache_path)


def scores_to_table(scores: list) -> pd.DataFrame:
    return pd.DataFrame.from_records(scores)


all_results = load_results().filter_models().filter_tasks()

max_model_size, min_model_size = get_model_size_range()


benchmarks = mteb.get_benchmarks()
all_tasks = [task.metadata.name for task in mteb.get_tasks()]
benchmark_to_name = {benchmark: benchmark.name for benchmark in benchmarks}
default_benchmark = mteb.get_benchmark("MTEB(Multilingual)")
default_scores = default_benchmark.get_scores(base_results=all_results)

benchmark_select = gr.Dropdown(
    [bench.name for bench in benchmarks],
    value=default_benchmark.name,
    label="Prebuilt Benchmarks",
    info="Select one of our expert-selected benchmarks from MTEB publications.",
)
lang_select = gr.Dropdown(
    get_languages(),
    value=[],
    multiselect=True,
    label="Language",
    info="Select langauges to include.",
)
type_select = gr.Dropdown(
    get_task_types(),
    value=[],
    multiselect=True,
    label="Task Type",
    info="Select task types to include.",
)
domain_select = gr.Dropdown(
    get_domains(),
    value=[],
    multiselect=True,
    label="Domain",
    info="Select domains to include.",
)
task_select = gr.Dropdown(
    all_tasks,
    value=[],
    multiselect=True,
    label="Task",
    info="Select specific tasks to include",
)
eval_split_select = gr.Dropdown(
    ["test", "dev", "train", "validation"],
    value=[],
    multiselect=True,
    label="Splits",
    info="Select splits to include in the scores (best left blank).",
)

with gr.Blocks(fill_width=True, theme=gr.themes.Base()) as app:
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
                #     model_types = gr.CheckboxGroup(
                #         ["Encoder", "Cross-Encoder", "Bi-Encoders"],
                #         value=["Encoder", "Cross-Encoder", "Bi-Encoders"],
                #         label="Allowed Model Types",
                #         interactive=True,
                #     )
                # with gr.Column():
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
    with gr.Group():
        benchmark_select.render()
        scores = gr.State(default_scores)
        benchmark = gr.State(default_benchmark.to_dict())
        with gr.Row():
            lang_select.render()
            type_select.render()
            domain_select.render()
            eval_split_select.render()
        with gr.Accordion("Add and remove tasks:", open=False):
            task_select.render()
    table = scores_to_table(default_scores)
    dataframe = gr.DataFrame(
        table,
        # datatype=["html"] + ["markdown"] * (len(table.columns) - 1),
    )

    @gr.on(
        inputs=[
            task_select,
            lang_select,
            type_select,
            domain_select,
            eval_split_select,
        ],
        outputs=[benchmark],
    )
    # @json_cache
    def update_benchmark(task_names, languages, types, domains, eval_splits):
        return {
            "task_names": task_names,
            "languages": languages,
            "task_types": types,
            "domains": domains,
            "eval_splits": eval_splits,
        }

    @gr.on(
        inputs=[benchmark_select],
        outputs=[
            task_select,
            lang_select,
            type_select,
            domain_select,
            eval_split_select,
        ],
    )
    def update_criteria(benchmark_name):
        benchmark = mteb.get_benchmark(benchmark_name)
        return (
            benchmark.task_names,
            benchmark.languages,
            benchmark.task_types,
            benchmark.domains,
            benchmark.eval_splits,
        )

    @gr.on(
        inputs=[benchmark],
        outputs=[scores],
    )
    def update_scores(benchmark):
        if "name" not in benchmark:
            benchmark["name"] = "temp"
        benchmark = Benchmark(**benchmark)
        return benchmark.get_scores(base_results=all_results, format="wide")

    @gr.on(
        inputs=[
            scores,
        ],
        outputs=[dataframe],
    )
    def update_table(scores):
        return scores_to_table(scores)
