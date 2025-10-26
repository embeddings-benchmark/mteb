from mteb.abstasks.audio.abs_task_zero_shot_classification import (
    AbsTaskAudioZeroshotClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class RavdessZeroshotClassification(AbsTaskAudioZeroshotClassification):
    metadata = TaskMetadata(
        name="RavdessZeroshot",
        description="Emotion classification Dataset. RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes neutral,calm, happy, sad, angry, fearful, surprise, and disgust expressions. These 8 emtoions also serve as labels for the dataset.",
        reference="https://huggingface.co/datasets/narad/ravdess",
        dataset={
            "path": "narad/ravdess",
            "revision": "2894394c52a8621bf8bb2e4d7c3b9cf77f6fa80e",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-03-01", "2018-03-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database ofal Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 180

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "this person is feeling neutral",
            "this person is feeling calm",
            "this person is feeling happy",
            "this person is feeling sad",
            "this person is feeling angry",
            "this person is feeling fearful",
            "this person is feeling disgust",
            "this person is feeling surprised",
        ]
