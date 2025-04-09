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
        eval_splits=["test"],  # Only test split is available for accented English
        eval_langs=["eng-Latn"],  # Using BCP-47 format
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Accent identification"],
        license="cc0-1.0",
        annotations_creators="found",
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
    label_column_name: str = "accent"
    samples_per_label: int = 30 # Approximate placeholder because value varies
    is_cross_validation: bool = False
    
    def dataset_transform(self):
        # Simple filtering for valid accent labels
        for split in self.dataset:
            # Filter out samples with "None" accent
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["accent"] != "None" and example["accent"] is not None
            )
            
            # Clean up accent labels (removing "en_" prefix for readability)
            self.dataset[split] = self.dataset[split].map(
                lambda example: {"accent_clean": example["accent"].replace("en_", "")}
            )
            # Use cleaned accent as label
            self.dataset[split] = self.dataset[split].rename_column("accent_clean", self.label_column_name) 