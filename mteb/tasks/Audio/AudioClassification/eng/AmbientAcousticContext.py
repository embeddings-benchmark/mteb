from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmbientAcousticContextClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="AmbientAcousticContext",
        description="The Ambient Acoustic Context dataset contains 1-second segments for activities that occur in a workplace setting.",
        reference="https://dl.acm.org/doi/10.1145/3379503.3403535",
        dataset={
            "path": "flwrlabs/ambient-acoustic-context",
            "revision": "8c77edafc0cad477055ec099c253c87b2b08e77a",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],  # Dataset only has train split
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),  # Paper publication date
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",  # As specified in dataset card
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{10.1145/3379503.3403535,
            author = {Park, Chunjong and Min, Chulhong and Bhattacharya, Sourav and Kawsar, Fahim},
            title = {Augmenting Conversational Agents with Ambient Acoustic Contexts},
            year = {2020},
            isbn = {9781450375160},
            publisher = {Association for Computing Machinery},
            address = {New York, NY, USA},
            url = {https://doi.org/10.1145/3379503.3403535},
            doi = {10.1145/3379503.3403535},
            booktitle = {22nd International Conference on Human-Computer Interaction with Mobile Devices and Services},
            articleno = {33},
            numpages = {9},
            keywords = {Acoustic ambient context, Conversational agents},
            location = {Oldenburg, Germany},
            series = {MobileHCI '20}
        }""",
        descriptive_stats={
            "n_samples": {"train": 70254},  # As mentioned in dataset card
            "n_classes": 24,  # From dataset viewer
            "sampling_rate": 16000,  # From data instances example
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 300  # Placeholder because value varies
    is_cross_validation: bool = False
