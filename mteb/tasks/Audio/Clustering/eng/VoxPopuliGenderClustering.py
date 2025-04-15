from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliGenderClustering(AbsTaskAudioClustering):
    label_column_name: str = "gender"

    metadata = TaskMetadata(
        name="VoxPopuliGenderClustering",
        description="Clustering speech samples by speaker gender (male/female) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "multilang",  # This selects the multilingual config
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],  # Focus on one language for clustering
        main_score="cluster_accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Clustering"],
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
                "train": 7600,
                "validation": 1755,  # 22.5% of of 7800 (english samples)
                "test": 1840,  # 23.5% of of 7800 (english samples)
            },
        },
    )

    audio_column_name: str = "audio"

    def dataset_transform(self, dataset):
        """Filter to keep only English samples in all splits."""
        # VoxPopuli language codes: 0 = English (en)
        ENGLISH_CODE = 0

        transformed_dataset = {}
        for split in dataset:
            # Get indices of English samples using numeric code
            english_indices = [
                i
                for i, lang_code in enumerate(dataset[split]["language"])
                if lang_code == ENGLISH_CODE
            ]

            # Select only English samples
            if english_indices:
                transformed_dataset[split] = dataset[split].select(english_indices)
            else:
                transformed_dataset[split] = dataset[split]

        return transformed_dataset
