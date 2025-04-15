from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliLanguageID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliLanguageID",
        description="Classification of speech samples into one of 5 European languages (English, German, French, Spanish, Polish) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "multilang",  # This selects the multilingual config/subset
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=[
            "eng-Latn",  # English
            "deu-Latn",  # German
            "fra-Latn",  # French
            "spa-Latn",  # Spanish
            "pol-Latn",  # Polish
        ],  # Using BCP-47 format for the 5 main languages
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Spoken Language Identification"],
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
                "test": 7800,  # Approximate count for the 5 main languages in test split (train and validation will not be included)
            },
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "language"
    samples_per_label: int = 50  # For balanced training
    is_cross_validation: bool = (
        True  # Use cross-validation since we only have test split
    )
    n_splits: int = 5  # Standard 5-fold cross-validation
