from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioZeroshotClassification import (
    AbsTaskAudioZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskAudioZeroshotClassification):
    metadata = TaskMetadata(
        name="SpeechCommandsZeroshot",
        description="Sound Classification/Keyword Spotting Dataset. This is a set of one-second .wav audio files, each containing a single spoken English word or background noise. These words are from a small set of commands, and are spoken by a variety of different speakers. With total of 30 labels",
        reference="https://huggingface.co/datasets/google/speech_commands",
        dataset={
            "path": "google/speech_commands",
            "name": "v0.01",
            "revision": "57ba463ab37e1e7845e0626539a6f6d0fcfbe64a",
        },
        type="AudioZeroshotClassification",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-07-07", "2018-07-13"),
        domains=[
            "Spoken"
        ],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Keyword Spotting"],
        license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-1804-03209,
  author       = {Pete Warden},
  title        = {Speech Commands: {A} Dataset for Limited-Vocabulary Speech Recognition},
  journal      = {CoRR},
  volume       = {abs/1804.03209},
  year         = {2018},
  url          = {http://arxiv.org/abs/1804.03209},
  eprinttype    = {arXiv},
  eprint       = {1804.03209},
  timestamp    = {Mon, 13 Aug 2018 16:48:32 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1804-03209.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
    }""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 2000

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "Yes",
            "No",
            "Up",
            "Down",
            "Left",
            "Right",
            "On",
            "Off",
            "Stop",
            "Go",
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
            "Bed",
            "Bird",
            "Cat",
            "Dog",
            "Happy",
            "House",
            "Marvin",
            "Sheila",
            "Tree",
            "Wow",
        ]
