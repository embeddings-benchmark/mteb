from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioZeroshotClassification import (
    AbsTaskAudioZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class RavdessZeroshotClassification(AbsTaskAudioZeroshotClassification):
    metadata = TaskMetadata(
        name="Ravdess_Zeroshot",
        description="Emotion classification Dataset.",
        reference="https://huggingface.co/datasets/narad/ravdess",
        dataset={
            "path": "narad/ravdess",
            "revision": "2894394c52a8621bf8bb2e4d7c3b9cf77f6fa80e",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-03-01", "2018-03-16"),
        domains=[
            "Spoken"
        ], 
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-3.0", 
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@article{10.1371/journal.pone.0196391,
            doi = {10.1371/journal.pone.0196391},
            author = {Livingstone, Steven R. AND Russo, Frank A.},
            journal = {PLOS ONE},
            publisher = {Public Library of Science},
            title = {The Ryerson Audio-Visual Database ofal Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
            year = {2018},
            month = {05},
            volume = {13},
            url = {https://doi.org/10.1371/journal.pone.0196391},
            pages = {1-35},
            number = {5},
    }""",
        descriptive_stats={
            "n_samples": {"train": 1440},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 180

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "this person is feeling neutral",
            "this person is feeling calm",
            "this person is feeling happy",
            "this person is feeling sad",
            "this person is feeling angry",
            "this person is feeling fearful",
            "this person is feeling disgust",
            "this person is feeling surprised",
        ]
