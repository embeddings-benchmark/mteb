from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

import mteb
from mteb import Benchmark


@dataclass
class MenuEntry:
    name: str | None
    benchmarks: list[Benchmark | MenuEntry]
    description: str | None = None
    open: bool = False


BENCHMARK_ENTRIES = [
    MenuEntry(
        name="Select Benchmark",
        description="",
        open=False,
        benchmarks=mteb.get_benchmarks(["MTEB(Multilingual, v2)", "MTEB(eng, v2)"])
        + [
            MenuEntry(
                "Image",
                mteb.get_benchmarks(
                    [
                        "MIEB(Multilingual)",
                        "MIEB(eng)",
                        "MIEB(lite)",
                        "MIEB(Img)",
                        "VisualDocumentRetrieval",
                    ]
                ),
            ),
            MenuEntry(
                "Domain-Specific",
                mteb.get_benchmarks(
                    [
                        "MTEB(Code, v1)",
                        "MTEB(Law, v1)",
                        "MTEB(Medical, v1)",
                        "ChemTEB",
                    ]
                ),
            ),
            MenuEntry(
                "Language-specific",
                mteb.get_benchmarks(
                    [
                        "MTEB(Europe, v1)",
                        "MTEB(Indic, v1)",
                        "MTEB(Scandinavian, v1)",
                        "MTEB(cmn, v1)",
                        "MTEB(deu, v1)",
                        "MTEB(fra, v1)",
                        "MTEB(jpn, v1)",
                        "MTEB(kor, v1)",
                        "MTEB(pol, v1)",
                        "MTEB(rus, v1)",
                        "MTEB(fas, v1)",
                    ]
                )
                + [MenuEntry("Other", mteb.get_benchmarks(["MTEB(eng, v1)"]))],
            ),
            MenuEntry(
                "Miscellaneous",  # All of these are retrieval benchmarks
                mteb.get_benchmarks(
                    [
                        "BEIR",
                        "BEIR-NL",
                        "NanoBEIR",
                        "BRIGHT",
                        "BRIGHT (long)",
                        "BuiltBench(eng)",
                        "CoIR",
                        "FollowIR",
                        "LongEmbed",
                        "MINERSBitextMining",
                        "RAR-b",
                    ]
                ),
            ),
        ],
    ),
]


def _create_button(
    i: int,
    benchmark: Benchmark,
    state: gr.State,
    label_to_value: dict[str, str],
    **kwargs,
):
    val = benchmark.name
    label = (
        benchmark.display_name if benchmark.display_name is not None else benchmark.name
    )
    label_to_value[label] = benchmark.name
    button = gr.Button(
        label,
        variant="secondary" if i != 0 else "primary",
        icon=benchmark.icon,
        key=f"{i}_button_{val}",
        elem_classes="text-white",
        **kwargs,
    )

    def _update_variant(state: str, label: str) -> gr.Button:
        if state == label_to_value[label]:
            return gr.Button(variant="primary")
        else:
            return gr.Button(variant="secondary")

    def _update_value(label: str) -> str:
        return label_to_value[label]

    state.change(_update_variant, inputs=[state, button], outputs=[button])
    button.click(_update_value, inputs=[button], outputs=[state])
    return button


def make_selector(
    entries: list[MenuEntry],
) -> tuple[gr.State, gr.Column]:
    if not entries:
        raise ValueError("No entries were specified, can't build selector.")
    label_to_value = {}

    with gr.Column() as column:
        state = gr.State("selector_state")
        i = 0

        for category_entry in entries:
            gr.Markdown(f"## {category_entry.name}")
            if category_entry.description:
                gr.Markdown(category_entry.description)

            for benchmarks_group in category_entry.benchmarks:
                if isinstance(benchmarks_group, Benchmark):  # level 0
                    button = _create_button(
                        i, benchmarks_group, state, label_to_value, size="md"
                    )
                    i += 1
                    continue

                with gr.Accordion(benchmarks_group.name, open=benchmarks_group.open):
                    for benchmark_entry in benchmarks_group.benchmarks:
                        if isinstance(benchmark_entry, Benchmark):  # level 1
                            button = _create_button(
                                i, benchmark_entry, state, label_to_value, size="sm"
                            )
                            i += 1
                            continue

                        with gr.Accordion(
                            benchmark_entry.name, open=benchmark_entry.open
                        ):
                            for minor_benchmarks in benchmark_entry.benchmarks:
                                if not isinstance(minor_benchmarks, Benchmark):
                                    raise TypeError(
                                        f"The leaderboard only support three layers of nesting. Expected Benchmark, got {type(minor_benchmarks)}."
                                    )

                                button = _create_button(
                                    i,
                                    minor_benchmarks,
                                    state,
                                    label_to_value,
                                    size="sm",
                                )
                                i += 1
                            continue

        return state, column


if __name__ == "__main__":
    with gr.Blocks() as b:
        selector = make_selector(BENCHMARK_ENTRIES)

    b.launch()
