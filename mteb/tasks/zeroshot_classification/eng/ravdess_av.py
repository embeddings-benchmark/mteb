from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
"""

DATASET = {
    "path": "mteb/RAVDESS_AV",
    "revision": "13af08387c3ce5e86c179a3718eb158669268d65",
}

DESCRIPTION_BASE = (
    "Emotion zero-shot classification on RAVDESS for 8 emotions: neutral, "
    "calm, happy, sad, angry, fearful, surprise, and disgust expressions."
)


def _emotion_prompts(names: list[str]) -> list[str]:
    return [f"this person is feeling {name}" for name in names]


class RAVDESSAVZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="RAVDESSAVZeroShot",
        description=DESCRIPTION_BASE + " This variant uses both video and audio. Used the metadata train split remapped to audio-visual filenames (~1,440 examples).",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset=DATASET,
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "emotion"

    def get_candidate_labels(self) -> list[str]:
        return _emotion_prompts(
            self.dataset["test"].features[self.label_column_name].names
        )


class RAVDESSVZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="RAVDESSVZeroShot",
        description=DESCRIPTION_BASE + " This variant uses video only. Used the metadata train split remapped to audio-visual filenames (~1,440 examples).",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset=DATASET,
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name: str = "video"
    label_column_name: str = "emotion"

    def get_candidate_labels(self) -> list[str]:
        return _emotion_prompts(
            self.dataset["test"].features[self.label_column_name].names
        )
