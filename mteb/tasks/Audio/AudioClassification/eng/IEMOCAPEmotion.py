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
        bibtex_citation="""@article{busso2008iemocap,
            title={IEMOCAP: Interactive emotional dyadic motion capture database},
            author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
            journal={Language resources and evaluation},
            volume={42},
            number={4},
            pages={335--359},
            year={2008},
            publisher={Springer}
        }""",
        descriptive_stats={
            "n_samples": {"train": 5000},  # Approximate after subsampling
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "major_emotion"
    samples_per_label: int = 50  # Approximate placeholder because value varies
    is_cross_validation: bool = False

    def dataset_transform(self):
        # Basic filtering to ensure we have valid emotion labels
        for split in self.dataset:
            # Ensure we have valid emotion labels and that they're normalized
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["major_emotion"] is not None
                and example["major_emotion"] != ""
            )

            # Map emotion labels to lowercase for consistency
            self.dataset[split] = self.dataset[split].map(
                lambda example: {
                    "major_emotion_clean": example["major_emotion"].lower()
                }
            )
            # Use cleaned emotion as label
            self.dataset[split] = self.dataset[split].rename_column(
                "major_emotion_clean", self.label_column_name
            )

            # Simple subsample if dataset is very large
            if len(self.dataset[split]) > 5000:
                self.dataset[split] = (
                    self.dataset[split].shuffle(seed=42).select(range(5000))
                )
