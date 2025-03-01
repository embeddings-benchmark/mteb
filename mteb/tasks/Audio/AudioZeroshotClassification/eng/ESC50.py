from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioZeroshotClassification import (
    AbsTaskAudioZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskAudioZeroshotClassification):
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
            "This is a sound of dog",
            "This is a sound of rooster",
            "This is a sound of pig",
            "This is a sound of cow",
            "This is a sound of frog",
            "This is a sound of cat",
            "This is a sound of hen",
            "This is a sound of insects",
            "This is a sound of sheep",
            "This is a sound of crow",
            "This is a sound of rain",
            "This is a sound of sea_waves",
            "This is a sound of crackling_fire",
            "This is a sound of crickets",
            "This is a sound of chirping_birds",
            "This is a sound of water_drops",
            "This is a sound of wind",
            "This is a sound of pouring_water",
            "This is a sound of toilet_flush",
            "This is a sound of thunderstorm",
            "This is a sound of crying_baby",
            "This is a sound of sneezing",
            "This is a sound of clapping",
            "This is a sound of breathing",
            "This is a sound of coughing",
            "This is a sound of footsteps",
            "This is a sound of laughing",
            "This is a sound of brushing_teeth",
            "This is a sound of snoring",
            "This is a sound of drinking_sipping",
            "This is a sound of door_wood_knock",
            "This is a sound of mouse_click",
            "This is a sound of keyboard_typing",
            "This is a sound of door_wood_creaks",
            "This is a sound of can_opening",
            "This is a sound of washing_machine",
            "This is a sound of vacuum_cleaner",
            "This is a sound of clock_alarm",
            "This is a sound of clock_tick",
            "This is a sound of glass_breaking",
            "This is a sound of helicopter",
            "This is a sound of chainsaw",
            "This is a sound of siren",
            "This is a sound of car_horn",
            "This is a sound of engine",
            "This is a sound of train",
            "This is a sound of church_bells",
            "This is a sound of airplane",
            "This is a sound of fireworks",
            "This is a sound of hand_saw",
        ]
