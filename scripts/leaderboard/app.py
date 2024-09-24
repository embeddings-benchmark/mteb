"""Notes:

Todo:
- [ ] Add model filtering
- [x] Add metadata column selection
- [ ] Add missing metadata columns
  - [ ] Model metadata (embedding size)
  - [ ] total Co2 emissions
- [ ] Add results loading from Hub
- [ ] Benchmark selection
- [ ] task type selection
- [ ] domain selection
- [ ] inidivual task selection

- Optimization to be added:
  - mteb.load_results is called for each custom language selection. A solution it so load the results once and filter them in the app.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

import mteb
import mteb.task_selection as task_selection

tasks = mteb.get_tasks()

languages = list(set(sum([task.languages for task in tasks], [])))
metadata_columns = ["Revision"]


class Default:
    languages: list[str] = []
    metadata_columns: list[str] = []


def get_mteb_results(languages: list[str] | None = None) -> pd.DataFrame:
    lang_str = "_".join(languages) if languages else "all"
    file_path: Path = Path(__file__).parent / f"results_{lang_str}.csv"

    tasks = mteb.get_tasks(languages=languages)
    if not file_path.exists():
        mteb_results = mteb.load_results(tasks=tasks)
        df = task_selection.results_to_dataframe(mteb_results, drop_na=False)
        df.to_csv(file_path)
    df = pd.read_csv(file_path)

    return df


def _update_dataframe(languages, metadata):
    _df = get_mteb_results(languages)
    cols_to_remove = [col for col in metadata_columns if col not in metadata]
    _df = _df.drop(columns=cols_to_remove)
    return _df


df = get_mteb_results()
df = _update_dataframe(Default.languages, Default.metadata_columns)


with gr.Blocks() as demo:
    with gr.Row():
        lang_select = gr.Dropdown(
            languages,
            value=[],
            multiselect=True,
            label="Language",
            info="Select langauges to filter by.",
        )
        metadata_select = gr.Dropdown(
            ["Revision"],
            value=[],
            multiselect=True,
            label="Metadata",
            info="Select model metadata columns to shown.",
        )

    dataframe = gr.DataFrame(df)

    @gr.on(inputs=[lang_select, metadata_select], outputs=dataframe)
    def update_dataframe(languages, metadata):
        return _update_dataframe(languages, metadata)


demo.launch()
