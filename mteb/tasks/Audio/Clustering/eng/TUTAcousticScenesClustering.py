from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class TUTAcousticScenesClustering(AbsTaskAudioClustering):
    label_column_name: str = "scene_label"
    metadata = TaskMetadata(
        name="TUTAcousticScenesClustering",
        description="Clustering task based on the TUT Urban Acoustic Scenes 2018 dataset with 10 different acoustic scenes.",
        reference="https://zenodo.org/record/1228142",
        dataset={
            "path": "adinmtn/TUT-urban-acoustic-scenes-2018-development",
            "revision": "583b181ea2666eb28d10909784690009f6c9da9d",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{Mesaros2018_DCASE,
            author = {Mesaros, Annamaria and Heittola, Toni and Virtanen, Tuomas},
            title = {A multi-device dataset for urban acoustic scene classification},
            booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
            year = {2018},
            pages = {9--13},
            publisher = {Tampere University of Technology},
            address = {Tampere, Finland}
        }""",
        descriptive_stats={
            "n_samples": {"train": 8640},  # Based on provided stats
            "n_classes": 10,
            "classes": [
                "airport",
                "bus",
                "metro",
                "metro_station",
                "park",
                "public_square",
                "shopping_mall",
                "street_pedestrian",
                "street_traffic",
                "tram",
            ],
        },
    )
