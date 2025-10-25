from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class MridinghamTonic(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="MridinghamTonic",
        description="Tonic classification of Mridingham (a pitched percussion instrument) into one of 6 classes: B,C,C#,D,D#,E",
        reference="https://huggingface.co/datasets/silky1708/Mridingham-Tonic",
        dataset={
            "path": "mteb/mridingham-tonic",
            "revision": "9304553355441e3d2bf2432691b6209ff9a9339c",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-10-06"),
        domains=["Music"],
        task_subtypes=["Tonic Classification of Musical Instrument"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{6637633,
  author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A},
  booktitle = {2013 IEEE International Conference on Acoustics, Speech and Signal Processing},
  doi = {10.1109/ICASSP.2013.6637633},
  keywords = {Instruments;Vectors;Hidden Markov models;Harmonic analysis;Modal analysis;Dictionaries;Music;Modal Analysis;Mridangam;automatic transcription;Non-negative Matrix Factorization;Hidden Markov models},
  number = {},
  pages = {181-185},
  title = {Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization},
  volume = {},
  year = {2013},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
