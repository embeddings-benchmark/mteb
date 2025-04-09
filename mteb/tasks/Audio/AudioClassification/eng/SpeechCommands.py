from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpeechCommandsClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SpeechCommands",
        description="A set of one-second .wav audio files, each containing a single spoken English word or background noise.",
        reference="https://arxiv.org/abs/1804.03209",
        dataset={
            "path": "google/speech_commands",
            "revision": "57ba463ab37e1e7845e0626539a6f6d0fcfbe64a",  # Using v0.02 as it's the latest version
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-04-11", "2018-04-11"),  # v0.02 release date
        domains=["Speech"],
        task_subtypes=["Spoken Digit Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@article{speechcommands2018,
            title={Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
            author={Pete Warden},
            journal={arXiv preprint arXiv:1804.03209},
            year={2018}
        }""",
        descriptive_stats={
            "n_samples": {"test": 4890},  # From v0.02
            "n_classes": 35,  # 35 classes in v0.02
            "classes": [
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "bed",
                "bird",
                "cat",
                "dog",
                "happy",
                "house",
                "marvin",
                "sheila",
                "tree",
                "wow",
                "backward",
                "forward",
                "follow",
                "learn",
                "visual",
            ],
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 50 # Rough guess/placeholder
    is_cross_validation: bool = False
