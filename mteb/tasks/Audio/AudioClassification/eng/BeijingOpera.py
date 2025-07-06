from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BeijingOpera(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="BeijingOpera",
        description="Audio classification of percussion instruments into one of 4 classes: `Bangu`, `Naobo`, `Daluo`, and `Xiaoluo`",
        reference="https://huggingface.co/datasets/silky1708/BeijingOpera",
        dataset={
            "path": "mteb/beijing-opera",
            "revision": "fed432f5ad94bc8d76c96d0ba05a38e805254281",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2014-01-01", "2014-12-31"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{6853981,
  author = {Tian, Mi and Srinivasamurthy, Ajay and Sandler, Mark and Serra, Xavier},
  booktitle = {2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/ICASSP.2014.6853981},
  keywords = {Decision support systems;Conferences;Acoustics;Speech;Speech processing;Time-frequency analysis;Beijing Opera;Onset Detection;Drum Transcription;Non-negative matrix factorization},
  number = {},
  pages = {2159-2163},
  title = {A study of instrument-wise onset detection in Beijing Opera percussion ensembles},
  volume = {},
  year = {2014},
}
""",
        descriptive_stats={
            "n_samples": {"train": 236},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
