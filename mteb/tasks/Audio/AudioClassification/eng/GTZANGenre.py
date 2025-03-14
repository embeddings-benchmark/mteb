from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTZANGenre(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="GTZANGenre",
        description="Music Genre Classification (10 classes)",
        reference="https://huggingface.co/datasets/silky1708/GTZAN-Genre",
        dataset={
            "path": "silky1708/GTZAN-Genre",
            "revision": "5efdda59d0d185bfe17ada9b54d233349d0e0168",
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
        bibtex_citation="""@ARTICLE{1021072,
            author={Tzanetakis, G. and Cook, P.},
            journal={IEEE Transactions on Speech and Audio Processing}, 
            title={Musical genre classification of audio signals}, 
            year={2002},
            volume={10},
            number={5},
            pages={293-302},
            keywords={Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
            doi={10.1109/TSA.2002.800560}}""",
        descriptive_stats={
            "n_samples": {"train": 1000},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
