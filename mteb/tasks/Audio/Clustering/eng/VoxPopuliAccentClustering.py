from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliAccentClustering(AbsTaskAudioClustering):
    label_column_name: str = "accent_id"

    metadata = TaskMetadata(
        name="VoxPopuliAccentClustering",
        description="Clustering English speech samples by non-native accent from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "en_accented",  # This explicitly selects the accented English config
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],  # Only test split is available for accented English
        eval_langs=["eng-Latn"],  # Using BCP-47 format
        main_score="cluster_accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Accent Clustering"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{wang-etal-2021-voxpopuli,
            title = "{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation",
            author = "Wang, Changhan  and
              Riviere, Morgane  and
              Lee, Ann  and
              Wu, Anne  and
              Talnikar, Chaitanya  and
              Haziza, Daniel  and
              Williamson, Mary  and
              Pino, Juan  and
              Dupoux, Emmanuel",
            booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
            month = aug,
            year = "2021",
            address = "Online",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.acl-long.80",
            doi = "10.18653/v1/2021.acl-long.80",
            pages = "993--1003",
        }""",
        descriptive_stats={
            "n_samples": {"test": 6900},
        },
    )

    audio_column_name: str = "audio"

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

        # Map accent codes to numeric IDs for clustering
        accent2id = {
            "en_nl": 0,  # Dutch
            "en_de": 1,  # German
            "en_cs": 2,  # Czech
            "en_pl": 3,  # Polish
            "en_fr": 4,  # French
            "en_hu": 5,  # Hungarian
            "en_fi": 6,  # Finnish
            "en_ro": 7,  # Romanian
            "en_sk": 8,  # Slovak
            "en_es": 9,  # Spanish
            "en_it": 10,  # Italian
            "en_et": 11,  # Estonian
            "en_lt": 12,  # Lithuanian
            "en_hr": 13,  # Croatian
            "en_sl": 14,  # Slovene
        }

        # Add accent_id based on accent code
        def add_accent_id(example):
            example["accent_id"] = accent2id[example["accent"]]
            return example

        test_data = test_data.map(add_accent_id)
        print(f"Mapped {len(accent2id)} accent codes to numeric IDs")

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
