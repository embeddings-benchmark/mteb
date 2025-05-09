from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliAccentID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliAccentID",
        description="Classification of English speech samples into one of 15 non-native accents from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "en_accented",  # This explicitly selects the accented English config
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],  # Using BCP-47 format
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Accent identification"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wang-etal-2021-voxpopuli,
  address = {Online},
  author = {Wang, Changhan  and
Riviere, Morgane  and
Lee, Ann  and
Wu, Anne  and
Talnikar, Chaitanya  and
Haziza, Daniel  and
Williamson, Mary  and
Pino, Juan  and
Dupoux, Emmanuel},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.80},
  month = aug,
  pages = {993--1003},
  publisher = {Association for Computational Linguistics},
  title = {{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
  url = {https://aclanthology.org/2021.acl-long.80},
  year = {2021},
}
""",
        descriptive_stats={
            "n_samples": {"test": 6900},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "accent"
    samples_per_label: int = 50
    is_cross_validation: bool = False

    def dataset_transform(self):
        # Split test into train (80%) and new test (20%)
        import random

        import numpy as np

        random.seed(42)
        dataset = self.dataset

        # Function to filter out corrupted or empty audio samples
        def is_valid_audio(example):
            # Check if audio array exists and is not empty
            if "audio" not in example or "array" not in example["audio"]:
                return False

            # Get the audio array
            audio_array = example["audio"]["array"]

            # Check if array is empty or too short (needs at least 10 samples for wav2vec2)
            if (
                audio_array is None or len(audio_array) < 500
            ):  # Minimum length to avoid kernel error
                return False

            # Check for NaN or Inf values
            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                return False

            return True

        # Filter test data to remove corrupted samples
        print("Filtering out corrupted audio samples...")
        test_data = dataset["test"]
        valid_indices = []

        # Find valid indices
        for i in range(len(test_data)):
            if is_valid_audio(test_data[i]):
                valid_indices.append(i)

        # Use only valid samples
        test_data = test_data.select(valid_indices)
        print(
            f"Kept {len(valid_indices)} valid samples out of {len(dataset['test'])} total"
        )

        # Continue with the original split logic
        indices = list(range(len(test_data)))
        random.shuffle(indices)

        split_point = int(len(indices) * 0.8)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        self.dataset = {
            "train": test_data.select(train_indices),
            "test": test_data.select(test_indices),
        }
        print(
            f"Created train split with {len(train_indices)} samples and test split with {len(test_indices)} samples"
        )
