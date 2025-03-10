from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTZANGenre(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="GTZANGenre",
        description="Music Genre Classification (10 classes)",
        reference="https://huggingface.co/datasets/sanchit-gandhi/gtzan",
        dataset={
            "path": "sanchit-gandhi/gtzan",
            "revision": "4bd857132cb0e731bef3ec68558e7acc0a85f144",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-06-23", "2023-06-23"),
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
    label_column_name: str = "genre"
    samples_per_label: int = 10
    is_cross_validation: bool = True
