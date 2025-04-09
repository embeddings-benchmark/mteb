from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class TUTAcousticScenesClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="TUTAcousticScenes",
        description="TUT Urban Acoustic Scenes 2018 dataset consists of 10-second audio segments from 10 acoustic scenes recorded in six European cities.",
        reference="https://zenodo.org/record/1228142",
        dataset={
            "path": "wetdog/TUT-urban-acoustic-scenes-2018-development",
            "revision": "583b181ea2666eb28d10909784690009f6c9da9d",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"), 
        domains=["Spoken"],  # A more appropriate domain for this task could be put when the domain list is updated
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{Mesaros2018_DCASE,
            author    = {Annamaria Mesaros and Toni Heittola and Tuomas Virtanen},
            title     = {A Multi-Device Dataset for Urban Acoustic Scene Classification},
            booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
            year      = {2018},
            publisher = {Tampere University of Technology},
            address   = {Tampere, Finland},
            url       = {https://arxiv.org/abs/1807.09840}
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

    audio_column_name: str = "audio"
    label_column_name: str = "scene_label"
    samples_per_label: int = 864 # Roughly 864 samples per label
    is_cross_validation: bool = False
