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
            "path": "AdnanElAssadi/speech_commands_small",
            "revision": "a59564b91bf0cfcf587e11c2603fe42bae21e5f0",  # Using downsampled version of v0.02
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-04-11", "2018-04-11"),  # v0.02 release date
        domains=["Speech"],
        task_subtypes=["Spoken Language Identification"],
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
            "n_samples": {"train": 1755, "validation": 9982, "test": 4890},
            "n_classes": 36,
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
                "_unknown_",  # (likely background noise or silent segments)
            ],
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 50
    is_cross_validation: bool = False
