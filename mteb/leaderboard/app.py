from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gradio as gr
from gradio_rangeslider import RangeSlider

import mteb
from mteb.leaderboard.table import scores_to_tables


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = mteb.load_results()
        all_results.to_disk(results_cache_path)
        return all_results
    else:
        return mteb.BenchmarkResults.from_disk(results_cache_path)


all_results = load_results().filter_models()

# Model sizes in million parameters
min_model_size, max_model_size = 8, 46703

benchmarks = mteb.get_benchmarks()

default_benchmark = mteb.get_benchmark("MTEB(Multilingual)")
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
        with gr.Column(scale=1):
            gr.Markdown(
                """
            ### Model Selection
            Select models to rank based on an assortment of criteria. 
            """,
            )
            with gr.Group():
                with gr.Row(elem_classes="overflow-y-scroll max-h-80"):
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
        with gr.Column(scale=2):
            gr.Markdown(
                """
            ### Benchmarks
            Select one of the hand-curated benchmarks from our publication.
            Or create one from scratch based on your use case.
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
    default_scores = default_results.get_scores(format="long")
    scores = gr.State(default_scores)
    summary, per_task = scores_to_tables(default_scores)
    with gr.Tab("Summary"):
        summary_table = gr.DataFrame(summary)
    with gr.Tab("Performance per task"):
        per_task_table = gr.DataFrame(per_task)

    @gr.on(inputs=[scores], outputs=[summary_table, per_task_table])
    def update_tables(scores):
        summary, per_task = scores_to_tables(scores)
        return summary, per_task

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
