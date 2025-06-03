from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class IEMOCAPEmotionClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="IEMOCAPEmotion",
        description="Classification of speech samples into emotions (angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted) from interactive emotional dyadic conversations.",
        reference="https://doi.org/10.1007/s10579-008-9076-6",
        dataset={
            "path": "AbstractTTS/IEMOCAP",
            "revision": "9f1696a135a65ce997d898d4121c952269a822ca",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2008-01-01", "2008-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{busso2008iemocap,
  author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal = {Language resources and evaluation},
  number = {4},
  pages = {335--359},
  publisher = {Springer},
  title = {IEMOCAP: Interactive emotional dyadic motion capture database},
  volume = {42},
  year = {2008},
}
""",
        descriptive_stats={
            "n_samples": {"train": 10039},  # Approximate after subsampling
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "emotion"
    samples_per_label: int = 10
    is_cross_validation: bool = True

    def dataset_transform(self):
        # Define emotion labels and their mapping to indices
        labels = [
            "angry",  # 0
            "sad",  # 1
            "happy",  # 2
            "neutral",  # 3
            "frustrated",  # 4
            "excited",  # 5
            "fear",  # 6
            "surprise",  # 7
            "disgust",  # 8
            "other",  # 9
        ]
        label2id = {emotion: idx for idx, emotion in enumerate(labels)}

        # Basic filtering to ensure we have valid emotion labels
        for split in self.dataset:
            # First ensure we have valid emotion labels and normalize case
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["major_emotion"] is not None
                and example["major_emotion"] != ""
            )

            # Map to indices with case normalization for reliability
            self.dataset[split] = self.dataset[split].map(
                lambda example: {
                    "emotion_id": label2id.get(example["major_emotion"].lower(), -1)
                }
            )

            # Filter out any examples with unknown emotions
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["emotion_id"] != -1
            )

            # Use numeric ID as the label
            self.dataset[split] = self.dataset[split].rename_column(
                "emotion_id", self.label_column_name
            )
