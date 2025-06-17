from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliGenderClustering(AbsTaskAudioClustering):
    label_column_name: str = "gender_id"

    metadata = TaskMetadata(
        name="VoxPopuliGenderClustering",
        description="Clustering speech samples by speaker gender (male/female) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "AdnanElAssadi/mini-voxpopuli",
            "revision": "70031eb5affcb0805e448fdf0b2dbbfc05f0aa8f",
            "trust_remote_code": True,
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Clustering"],
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
            "n_samples": {
                "train": 7600,
                "validation": 1750,
                "test": 1840,
            },
        },
    )

    audio_column_name: str = "audio"

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
