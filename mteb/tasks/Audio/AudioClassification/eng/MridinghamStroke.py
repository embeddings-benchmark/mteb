from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MridinghamStroke(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="MridinghamStroke",
        description='Stroke classification of Mridingham (a pitched percussion instrument) into one of 10 classes: ["bheem", "cha", "dheem", "dhin", "num", "tham", "ta", "tha", "thi", "thom"]',
        reference="https://huggingface.co/datasets/silky1708/Mridingham-Stroke",
        dataset={
            "path": "silky1708/Mridingham-Stroke",
            "revision": "523fe0aac393bbd1a9b46a77951d09296a1b4932",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-10-06"),
        domains=["Music"],
        task_subtypes=["Stroke Classification of Musical Instrument"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@INPROCEEDINGS{6637633,
            author={Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A},
            booktitle={2013 IEEE International Conference on Acoustics, Speech and Signal Processing}, 
            title={Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization}, 
            year={2013},
            volume={},
            number={},
            pages={181-185},
            keywords={Instruments;Vectors;Hidden Markov models;Harmonic analysis;Modal analysis;Dictionaries;Music;Modal Analysis;Mridangam;automatic transcription;Non-negative Matrix Factorization;Hidden Markov models},
            doi={10.1109/ICASSP.2013.6637633}}
        """,
        descriptive_stats={
            "n_samples": {"train": 6977},  # test samples not found!
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
