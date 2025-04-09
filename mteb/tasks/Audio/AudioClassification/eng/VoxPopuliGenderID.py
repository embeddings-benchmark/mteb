from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliGenderID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliGenderID",
        description="Classification of speech samples by speaker gender (male/female) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "multilang",  # This explicitly selects the multilingual config
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],  # Focus on English for simplicity and comparability
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
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
            "n_samples": {
                "train": 5000,
                "validation": 500,
                "test": 500,
            },  # Approx after filtering to English
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "gender"
    samples_per_label: int = 50  # Approximate placeholder because value varies
    is_cross_validation: bool = False

    def dataset_transform(self):
        # Filter out samples with missing gender information and focus on English language
        for split in self.dataset:
            # Filter to get only English (language 0) with valid gender labels
            self.dataset[split] = self.dataset[split].filter(
                lambda example: (
                    example["gender"] is not None
                    and example["gender"] in ["male", "female"]
                    and example["language"] == 0
                )  # 0 is English
            )

            # Simple subsample if dataset is very large (optional)
            if len(self.dataset[split]) > 1000:
                self.dataset[split] = self.dataset[split].select(range(1000))
