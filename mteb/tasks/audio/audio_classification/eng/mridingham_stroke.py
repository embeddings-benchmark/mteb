from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class MridinghamStroke(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="MridinghamStroke",
        description='Stroke classification of Mridingham (a pitched percussion instrument) into one of 10 classes: ["bheem", "cha", "dheem", "dhin", "num", "tham", "ta", "tha", "thi", "thom"]',
        reference="https://huggingface.co/datasets/silky1708/Mridingham-Stroke",
        dataset={
            "path": "mteb/mridingham-stroke",
            "revision": "2b6d06ff3da6da9609f078b6bc92c248016a5112",
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
