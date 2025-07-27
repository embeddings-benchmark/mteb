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


def make_selector(entries: list[MenuEntry]) -> tuple[gr.State, gr.Column]:
    """Creates a UI selector from menu entries with up to 3 levels of nesting.

    Args:
        entries: List of MenuEntry objects to build the selector from

    Returns:
        tuple: (state object, column widget)
    """
    label_to_value = {}
    button_counter = 0

    with gr.Column() as column:
        state = gr.State("selector_state")

        for category_entry in entries:
            button_counter = _render_category(
                category_entry, state, label_to_value, button_counter
            )

    return state, column


def _render_category(
    entry: MenuEntry,
    state: gr.State,
    label_to_value: dict,
    button_counter: int,
) -> int:
    gr.Markdown(f"## {entry.name}")
    if entry.description:
        gr.Markdown(entry.description)

    for benchmarks_group in entry.benchmarks:
        button_counter = _render_benchmark_item(
            benchmarks_group, state, label_to_value, button_counter, level=0
        )

    return button_counter


def _render_benchmark_item(
    item: Benchmark | MenuEntry,
    state: gr.State,
    label_to_value: dict,
    button_counter: int,
    level: int,
) -> int:
    if isinstance(item, Benchmark):
        size = "md" if level == 0 else "sm"
        _create_button(button_counter, item, state, label_to_value, size=size)
        return button_counter + 1

    with gr.Accordion(item.name, open=item.open):
        for nested_item in item.benchmarks:
            button_counter = _render_benchmark_item(
                nested_item, state, label_to_value, button_counter, level + 1
            )

    return button_counter


if __name__ == "__main__":
    with gr.Blocks() as b:
        selector = make_selector(BENCHMARK_ENTRIES)

    b.launch()
