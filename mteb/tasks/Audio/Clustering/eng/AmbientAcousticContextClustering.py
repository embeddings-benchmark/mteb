from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmbientAcousticContextClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"

    metadata = TaskMetadata(
        name="AmbientAcousticContextClustering",
        description="Clustering task based on a subset of the Ambient Acoustic Context dataset containing 1-second segments for workplace activities.",
        reference="https://dl.acm.org/doi/10.1145/3379503.3403535",
        dataset={
            "path": "AdnanElAssadi/ambient-acoustic-context-small",
            "revision": "360c858462b79492c6b09d5855ec4d59c87497c6",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Clustering"],
        license="not specified",
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
            "n_samples": {
                "train": 2387,  # ~100 samples × 24 classes
                "test": 1036,  # ~50 samples × 24 classes
            },
            "n_classes": 24,
            "sampling_rate": 16000,
        },
    )
