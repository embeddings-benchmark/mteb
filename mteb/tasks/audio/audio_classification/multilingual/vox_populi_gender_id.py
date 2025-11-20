from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class VoxPopuliGenderID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliGenderID",
        description="Subsampled Dataset Classification of speech samples by speaker gender (male/female) from European Parliament recordings.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "mteb/voxpopuli-mini",
            "revision": "d5fb9661054ba250e2c03adeb9a702ad55e73f27",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=[
            "eng-Latn",  # English
            "fra-Latn",  # French
            "spa-Latn",  # Spanish
            "pol-Latn",  # Polish
            "deu-Latn",  # German
        ],
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
    )

    audio_column_name: str = "audio"
    label_column_name: str = "gender_id"
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
