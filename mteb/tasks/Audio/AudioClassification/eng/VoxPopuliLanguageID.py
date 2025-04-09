from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliLanguageID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliLanguageID",
        description="Classification of speech samples into one of 18 European languages from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "multilang",  # This explicitly selects the multilingual config
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn", "deu-Latn", "fra-Latn", "spa-Latn", "pol-Latn", 
                   "ita-Latn", "ron-Latn", "hun-Latn", "ces-Latn", "nld-Latn"],  # Using BCP-47 format
        main_score="accuracy",
        date=("2009-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Language Identification"],
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
            "n_samples": {"train": 50000, "validation": 5000, "test": 5000},  # Approx
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "language"
    samples_per_label: int = 30 # Approximate placeholder because value varies
    is_cross_validation: bool = False
    
    def dataset_transform(self):
        # Convert numerical language IDs to string language codes for better interpretability
        language_map = {
            0: "en", 1: "de", 2: "fr", 3: "es", 4: "pl", 5: "it", 6: "ro",
            7: "hu", 8: "cs", 9: "nl", 10: "fi", 11: "hr", 12: "sk",
            13: "sl", 14: "et", 15: "lt", 16: "lv", 17: "da"
        }
        
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda example: {"language_code": language_map[example["language"]]}
            )
            # Use language code as label
            self.dataset[split] = self.dataset[split].rename_column("language_code", self.label_column_name)
            
            # Simple subsample if dataset is very large (optional)
            if len(self.dataset[split]) > 5000:
                self.dataset[split] = self.dataset[split].select(range(5000)) 