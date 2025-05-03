from __future__ import annotations

import gradio as gr

import mteb
from mteb import Benchmark

"""
Each entry is a tuple, where the first element is a label, and the second is either a single benchmark or a group of benchmarks.

Example:
[
    ("First Benchmark", dict(value="MTEB(something)", icon="icon_url")),
    ("Group of Benchmarks",
        [
            ("Second Benchmark", dict(value="MTEB(something)", icon="icon_url")),
            ("Third Benchmark", dict(value="MTEB(something)", icon="icon_url")),
        ],
    ),
]
"""
BENCHMARK_ENTRIES = [
    mteb.get_benchmarks(["MTEB(Multilingual, v2)", "MTEB(eng, v2)"]),
    (
        "Image Benchmarks",
        mteb.get_benchmarks(
            [
                "MIEB(Multilingual)",
                "MIEB(eng)",
                "MIEB(lite)",
                "MIEB(Img)",
            ]
        ),
    ),
    (
        "Domain-Specific Benchmarks",
        mteb.get_benchmarks(
            [
                "MTEB(Code, v1)",
                "MTEB(Law, v1)",
                "MTEB(Medical, v1)",
                "ChemTEB",
            ]
        ),
    ),
    (
        "Regional Benchmarks",
        mteb.get_benchmarks(
            [
                "MTEB(Europe, v1)",
                "MTEB(Indic, v1)",
                "MTEB(Scandinavian, v1)",
            ]
        ),
    ),
    (
        "Language-specific Benchmarks",
        mteb.get_benchmarks(
            [
                "MTEB(cmn, v1)",
                "MTEB(deu, v1)",
                "MTEB(fra, v1)",
                "MTEB(jpn, v1)",
                "MTEB(kor, v1)",
                "MTEB(pol, v1)",
                "MTEB(rus, v1)",
                "MTEB(fas, beta)",
            ]
        ),
    ),
    (
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
    (
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
    entries: list[list[Benchmark] | tuple[str, list[Benchmark]]],
) -> tuple[gr.State, gr.Column]:
    if not entries:
        raise ValueError("No entries were specified, can't build selector.")
    label_to_value = {}
    state = None
    with gr.Column() as column:
        i = 0
        for entry in entries:
            if i == 0:
                if isinstance(entry, list):
                    fist_entry = entry[0]
                    state = gr.State(fist_entry.name)
                elif isinstance(entry, tuple):
                    _label, _entry = entry
                    state = gr.State(_entry[0].name)
                else:
                    raise ValueError("Benchmark selector specified incorrectly")
            if isinstance(entry, list):
                for benchmark in entry:
                    button = _create_button(
                        i, benchmark, state, label_to_value, size="lg"
                    )
                    i += 1
            elif isinstance(entry, tuple):
                label, _entry = entry
                gr.Markdown(f"### **{label}**")
                for benchmark in _entry:
                    button = _create_button(  # noqa: F841
                        i, benchmark, state, label_to_value, size="md"
                    )
                    i += 1

    return state, column


if __name__ == "__main__":
    with gr.Blocks() as b:
        selector = make_selector(BENCHMARK_ENTRIES)

    b.launch()
