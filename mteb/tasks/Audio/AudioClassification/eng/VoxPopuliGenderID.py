from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

EVAL_LANGS_MAP = {
    "en-GB": ["eng-Latn"],  # English
    "fr-FR": ["fra-Latn"],  # French
    "es-ES": ["spa-Latn"],  # Spanish
    "pl-PL": ["pol-Latn"],  # Polish
    "de-DE": ["deu-Latn"],  # German
}


class VoxPopuliGenderID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliGenderID",
        description="Classification of speech samples by speaker gender (male/female) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "AdnanElAssadi/mini-voxpopuli",
            "revision": "70031eb5affcb0805e448fdf0b2dbbfc05f0aa8f",
            "trust_remote_code": True,
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=EVAL_LANGS_MAP,
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
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
            "n_samples": {"train": 500},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "gender"
    samples_per_label: int = 30
    is_cross_validation: bool = True

    def dataset_transform(self):
        # Define label mapping
        label2id = {"female": 0, "male": 1}

        # Apply transformation to all dataset splits
        for split in self.dataset:
            # Define transform function to add numeric labels
            def add_gender_id(example):
                example["gender_id"] = label2id[example["gender"]]
                return example

            print(f"Converting gender labels to numeric IDs for split '{split}'...")
            self.dataset[split] = self.dataset[split].map(add_gender_id)
