from __future__ import annotations

import gradio as gr

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
    (
        "Multilingual",
        dict(
            value="MTEB(Multilingual, v1)",
            icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-globe.svg",
        ),
    ),
    (
        "English",
        dict(
            value="MTEB(eng, v2)",
            icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
        ),
    ),
    (
        "Image Benchmarks",
        [
            (
                "Images, Multilingual",
                dict(
                    value="MIEB(Multilingual)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-pictures.svg",
                ),
            ),
            (
                "Images, English",
                dict(
                    value="MIEB(eng)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-picture.svg",
                ),
            ),
            (
                "Images, Lite",
                dict(
                    value="MIEB(lite)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-landscape.svg",
                ),
            ),
        ],
    ),
    (
        "Domain-Specific Benchmarks",
        [
            (
                "Code",
                dict(
                    value="MTEB(Code, v1)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
                ),
            ),
            (
                "Legal",
                dict(
                    value="MTEB(Law, v1)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-library.svg",
                ),
            ),
            (
                "Medical",
                dict(
                    value="MTEB(Medical, v1)",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-hospital.svg",
                ),
            ),
            (
                "Chemical",
                dict(
                    value="ChemTEB",
                    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-purge.svg",
                ),
            ),
        ],
    ),
    (
        "Regional Benchmarks",
        [
            (
                "European",
                dict(
                    value="MTEB(Europe, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/eu.svg",
                ),
            ),
            (
                "Indic",
                dict(
                    value="MTEB(Indic, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/in.svg",
                ),
            ),
            (
                "Scandinavian",
                dict(
                    value="MTEB(Scandinavian, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/dk.svg",
                ),
            ),
        ],
    ),
    (
        "Language-specific Benchmarks",
        [
            (
                "Chinese",
                dict(
                    value="MTEB(cmn, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/cn.svg",
                ),
            ),
            (
                "German",
                dict(
                    value="MTEB(deu, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/de.svg",
                ),
            ),
            (
                "French",
                dict(
                    value="MTEB(fra, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/fr.svg",
                ),
            ),
            (
                "Japanese",
                dict(
                    value="MTEB(jpn, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
                ),
            ),
            (
                "Korean",
                dict(
                    value="MTEB(kor, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/kr.svg",
                ),
            ),
            (
                "Polish",
                dict(
                    value="MTEB(pol, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/pl.svg",
                ),
            ),
            (
                "Russian",
                dict(
                    value="MTEB(rus, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ru.svg",
                ),
            ),
            (
                "Farsi (BETA)",
                dict(
                    value="MTEB(fas, beta)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ir.svg",
                ),
            ),
        ],
    ),
    (
        "Miscellaneous",
        [
            ("BEIR", dict(value="BEIR", icon=None)),
            ("BEIR-NL", dict(value="BEIR-NL", icon=None)),
            ("BRIGHT", dict(value="BRIGHT", icon=None)),
            ("BRIGHT (long)", dict(value="BRIGHT (long)", icon=None)),
            ("BuiltBench (eng)", dict(value="BuiltBench(eng)", icon=None)),
            ("Code Information Retrieval", dict(value="CoIR", icon=None)),
            ("Instruction Following", dict(value="FollowIR", icon=None)),
            ("Long-context Retrieval", dict(value="LongEmbed", icon=None)),
            ("MINERSBitextMining", dict(value="MINERSBitextMining", icon=None)),
            ("NanoBEIR", dict(value="NanoBEIR", icon=None)),
            ("Reasoning retrieval", dict(value="RAR-b", icon=None)),
        ],
    ),
    (
        "Legacy",
        [
            (
                "English Legacy",
                dict(
                    value="MTEB(eng, v1)",
                    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/gb.svg",
                ),
            ),
        ],
    ),
]


def _create_button(i, label, entry, state, label_to_value, **kwargs):
    val = entry["value"]
    label_to_value[label] = val
    button = gr.Button(
        label,
        variant="secondary" if i != 0 else "primary",
        icon=entry["icon"],
        key=f"{i}_button_{val}",
        elem_classes="text-white",
        **kwargs,
    )

    def _update_variant(state, label) -> gr.Button:
        if state == label_to_value[label]:
            return gr.Button(variant="primary")
        else:
            return gr.Button(variant="secondary")

    def _update_value(label) -> str:
        return label_to_value[label]

    state.change(_update_variant, inputs=[state, button], outputs=[button])
    button.click(_update_value, outputs=[state], inputs=[button])
    return button


def make_selector(entries: list[tuple[str, dict | list]]) -> tuple[gr.State, gr.Column]:
    if not entries:
        raise ValueError("No entries were specified, can't build selector.")
    label_to_value = {}
    state = None
    with gr.Column() as column:
        for i, (label, entry) in enumerate(entries):
            if i == 0:
                if isinstance(entry, dict):
                    state = gr.State(entry["value"])
                else:
                    _label, _entry = entry[0]
                    state = gr.State(_entry["value"])
            if isinstance(entry, dict):
                button = _create_button(
                    i, label, entry, state, label_to_value, size="lg"
                )
            else:
                gr.Markdown(f"### **{label}**")
                for sub_label, sub_entry in entry:
                    button = _create_button(  # noqa: F841
                        i, sub_label, sub_entry, state, label_to_value, size="md"
                    )

    return state, column
