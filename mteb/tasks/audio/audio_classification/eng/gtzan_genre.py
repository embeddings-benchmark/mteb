from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class GTZANGenre(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="GTZANGenre",
        description="Music Genre Classification (10 classes)",
        reference="https://huggingface.co/datasets/silky1708/GTZAN-Genre",
        dataset={
            "path": "mteb/gtzan-genre",
            "revision": "53efa094d18619a4e2bf36123192ec7a01a7d1bf",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2001-12-31"),
        domains=["Music"],
        task_subtypes=["Music Genre Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{1021072,
  author = {Tzanetakis, G. and Cook, P.},
  doi = {10.1109/TSA.2002.800560},
  journal = {IEEE Transactions on Speech and Audio Processing},
  keywords = {Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
  number = {5},
  pages = {293-302},
  title = {Musical genre classification of audio signals},
  volume = {10},
  year = {2002},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
