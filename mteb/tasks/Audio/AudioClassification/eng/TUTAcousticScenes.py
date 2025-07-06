from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class TUTAcousticScenesClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="TUTAcousticScenes",
        description="TUT Urban Acoustic Scenes 2018 dataset consists of 10-second audio segments from 10 acoustic scenes recorded in six European cities. This is a stratified subsampled version of the original dataset.",
        reference="https://zenodo.org/record/1228142",
        dataset={
            "path": "mteb/tut-acoustic-scenes-mini",
            "revision": "fe74de34b726995a39971faeb83491480ca1886e",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=[
            "AudioScene",
        ],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Mesaros2018_DCASE,
  address = {Tampere, Finland},
  author = {Annamaria Mesaros and Toni Heittola and Tuomas Virtanen},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
  publisher = {Tampere University of Technology},
  title = {A Multi-Device Dataset for Urban Acoustic Scene Classification},
  url = {https://arxiv.org/abs/1807.09840},
  year = {2018},
}
""",
        descriptive_stats={
            "n_samples": {"train": 2000},  # Stratified subsampled from original 8640
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
    samples_per_label: int = 50
    is_cross_validation: bool = True
    n_splits: int = 5
