from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxPopuliAccentID(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxPopuliAccentID",
        description="Classification of English speech samples into one of 15 non-native accents from European Parliament recordings. This is a stratified subsampled version of the original VoxPopuli dataset.",
        reference="https://huggingface.co/datasets/facebook/voxpopuli",
        dataset={
            "path": "mteb/voxpopuli-accent-mini",
            "revision": "5ea6ce48d0de6db179cb7debf0d6813faa495a33",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
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
            "n_samples": {"test": 2000},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "accent"
    samples_per_label: int = 50
    is_cross_validation: bool = True

    def dataset_transform(self):
        import numpy as np
        from datasets import DatasetDict

        test_ds = self.dataset["test"]

        def is_valid_audio(example):
            audio_arr = example.get("audio", {}).get("array", None)
            # require at least 500 samples (so that Kaldi fbank(window_size=400) won't fail)
            if (audio_arr is None) or (len(audio_arr) < 500):
                return False
            if np.isnan(audio_arr).any() or np.isinf(audio_arr).any():
                return False
            return True

        filtered_test = test_ds.filter(is_valid_audio)
        print(f"Kept {len(filtered_test)} valid samples out of {len(test_ds)} total")

        # Create a new DatasetDict that has both "train" and "test" = filtered_test
        self.dataset = DatasetDict(
            {
                "train": filtered_test,
                "test": filtered_test,
            }
        )
