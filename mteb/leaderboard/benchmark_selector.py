from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

import mteb
from mteb import Benchmark


@dataclass
class MenuEntry:
    name: str | None
    benchmarks: list[Benchmark]
    open: bool = False
    size: str = "sm"


BENCHMARK_ENTRIES = [
    MenuEntry(
        None,
        mteb.get_benchmarks(["MTEB(Multilingual, v2)", "MTEB(eng, v2)"]),
        False,
        size="md",
    ),
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
        True,
    ),
    MenuEntry(
        "Regional",
        mteb.get_benchmarks(
            [
                "MTEB(Europe, v1)",
                "MTEB(Indic, v1)",
                "MTEB(Scandinavian, v1)",
            ]
        ),
        True,
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
                "MTEB(cmn, v1)",
                "MTEB(deu, v1)",
                "MTEB(fra, v1)",
                "MTEB(jpn, v1)",
                "MTEB(kor, v1)",
                "MTEB(pol, v1)",
                "MTEB(rus, v1)",
                "MTEB(fas, v1)",
            ]
        ),
    ),
    MenuEntry(
        "Miscellaneous",
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
    MenuEntry(
        "Legacy",
        mteb.get_benchmarks(
            [
                "MTEB(eng, v1)",
            ]
        ),
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
        state = gr.State(entries[0].benchmarks[0].name)
        i = 0
        for entry in entries:
            if entry.name is None:
                for benchmark in entry.benchmarks:
                    button = _create_button(
                        i, benchmark, state, label_to_value, size=entry.size
                    )
                    i += 1
            if entry.name is not None:
                with gr.Accordion(entry.name, open=entry.open):
                    for benchmark in entry.benchmarks:
                        button = _create_button(  # noqa: F841
                            i, benchmark, state, label_to_value, size=entry.size
                        )
                        i += 1

    return state, column


if __name__ == "__main__":
    with gr.Blocks() as b:
        selector = make_selector(BENCHMARK_ENTRIES)

    b.launch()
