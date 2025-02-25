from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskZeroshotClassificaiton import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="ESC50_Zeroshot",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "ashraq/esc50",
            "revision": "e3e2a63ffff66b9a9735524551e3818e96af03ee",
        },
        type="AudioZeroshotClassification",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-01-07", "2023-01-07"),
        domains=[
            "Spoken"
        ],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{piczak2015dataset,
        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
        author = {Piczak, Karol J.},
        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
        date = {2015-10-13},
        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
        doi = {10.1145/2733373.2806390},
        location = {{Brisbane, Australia}},
        isbn = {978-1-4503-3459-4},
        publisher = {{ACM Press}},
        pages = {1015--1018}
    }""",
        descriptive_stats={
            "n_samples": {"train": 2000},  # Need actual number
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "target"
    samples_per_label: int = 50

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "dog",
            "rooster",
            "pig",
            "cow",
            "frog",
            "cat",
            "hen",
            "insects",
            "sheep",
            "crow",
            "rain",
            "sea_waves",
            "crackling_fire",
            "crickets",
            "chirping_birds",
            "water_drops",
            "wind",
            "pouring_water",
            "toilet_flush",
            "thunderstorm",
            "crying_baby",
            "sneezing",
            "clapping",
            "breathing",
            "coughing",
            "footsteps",
            "laughing",
            "brushing_teeth",
            "snoring",
            "drinking_sipping",
            "door_wood_knock",
            "mouse_click",
            "keyboard_typing",
            "door_wood_creaks",
            "can_opening",
            "washing_machine",
            "vacuum_cleaner",
            "clock_alarm",
            "clock_tick",
            "glass_breaking",
            "helicopter",
            "chainsaw",
            "siren",
            "car_horn",
            "engine",
            "train",
            "church_bells",
            "airplane",
            "fireworks",
            "hand_saw",
        ]
