from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

import mteb
from mteb import Benchmark
from mteb.benchmarks.benchmarks import MTEB_multilingual_v2

DEFAULT_BENCHMARK_NAME = MTEB_multilingual_v2.name


@dataclass
class MenuEntry:
    """A menu entry for the benchmark selector.

    Attributes:
        name: The name of the menu entry.
        benchmarks: A list of benchmarks or nested menu entries.
        description: An optional description of the menu entry.
        open: Whether the accordion is open by default.
        size: The size of the buttons. Can be "sm" or "md".
    """

    name: str | None
    benchmarks: list[Benchmark | MenuEntry]
    description: str | None = None
    open: bool = False
    size: str = "sm"


GP_BENCHMARK_ENTRIES = [
    MenuEntry(
        name="General Purpose",
        description="",
        open=False,
        benchmarks=mteb.get_benchmarks(
            ["MTEB(Multilingual, v2)", "MTEB(eng, v2)", "HUME(v1)"]
        )
        + [
            MenuEntry(
                "Image",
                mteb.get_benchmarks(
                    [
                        "MIEB(Multilingual)",
                        "MIEB(eng)",
                        "MIEB(lite)",
                        "MIEB(Img)",
                    ]
                ),
            ),
            MenuEntry(
                "Domain-Specific ",
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
                        "JMTEB(v2)",
                        "MTEB(kor, v1)",
                        "MTEB(nld, v1)",
                        "MTEB(pol, v1)",
                        "MTEB(rus, v1.1)",
                        "MTEB(fas, v2)",
                        "VN-MTEB (vie, v1)",
                    ]
                )
                + [
                    MenuEntry(
                        "Other",
                        mteb.get_benchmarks(
                            [
                                "MTEB(eng, v1)",
                                "MTEB(fas, v1)",
                                "MTEB(rus, v1)",
                                "MTEB(jpn, v1)",
                            ]
                        ),
                    )
                ],
            ),
            MenuEntry(
                "Miscellaneous",  # All of these are retrieval benchmarks
                mteb.get_benchmarks(
                    [
                        "BuiltBench(eng)",
                        "MINERSBitextMining",
                    ]
                ),
            ),
        ],
    ),
]

R_BENCHMARK_ENTRIES = [
    MenuEntry(
        name="Retrieval",
        description=None,
        open=False,
        benchmarks=[
            mteb.get_benchmark("RTEB(beta)"),
            mteb.get_benchmark("RTEB(eng, beta)"),
            MenuEntry(
                "Image",
                description=None,
                open=True,
                benchmarks=[
                    mteb.get_benchmark("ViDoRe(v3)"),
                    mteb.get_benchmark("JinaVDR"),
                    MenuEntry("Other", [mteb.get_benchmark("ViDoRe(v1&v2)")]),
                ],
            ),
            MenuEntry(
                "Domain-Specific",
                description=None,
                open=False,
                benchmarks=[
                    mteb.get_benchmark("RTEB(fin, beta)"),
                    mteb.get_benchmark("RTEB(Law, beta)"),
                    mteb.get_benchmark("RTEB(Code, beta)"),
                    mteb.get_benchmark("CoIR"),
                    mteb.get_benchmark("RTEB(Health, beta)"),
                    mteb.get_benchmark("FollowIR"),
                    mteb.get_benchmark("LongEmbed"),
                    mteb.get_benchmark("BRIGHT"),
                ],
            ),
            MenuEntry(
                "Language-specific",
                description=None,
                open=False,
                benchmarks=[
                    mteb.get_benchmark("RTEB(fra, beta)"),
                    mteb.get_benchmark("RTEB(deu, beta)"),
                    mteb.get_benchmark("RTEB(jpn, beta)"),
                    mteb.get_benchmark("BEIR"),
                    mteb.get_benchmark("BEIR-NL"),
                ],
            ),
            MenuEntry(
                "Miscellaneous",
                mteb.get_benchmarks(
                    [
                        "NanoBEIR",
                        "BRIGHT (long)",
                        "RAR-b",
                    ]
                ),
            ),
        ],
    )
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


def _make_selector(entries: list[MenuEntry]) -> tuple[gr.State, gr.Column]:
    """Creates a UI selector from menu entries with up to 3 levels of nesting.

    Args:
        entries: List of MenuEntry objects to build the selector from

    Returns:
        tuple: (state object, column widget)
    """
    label_to_value = {}
    button_counter = 0

    with gr.Column() as column:
        state = gr.State(DEFAULT_BENCHMARK_NAME)

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
        selector = _make_selector(GP_BENCHMARK_ENTRIES)
    b.launch()
