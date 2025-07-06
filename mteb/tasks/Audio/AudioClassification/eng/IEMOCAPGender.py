from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class IEMOCAPGenderClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="IEMOCAPGender",
        description="Classification of speech samples by speaker gender (male/female) from the IEMOCAP database of interactive emotional dyadic conversations.",
        reference="https://doi.org/10.1007/s10579-008-9076-6",
        dataset={
            "path": "mteb/iemocap",
            "revision": "6d4225271da423e791e16770d335cfa351cdf88e",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2008-01-01", "2008-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{busso2008iemocap,
  author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal = {Language resources and evaluation},
  number = {4},
  pages = {335--359},
  publisher = {Springer},
  title = {IEMOCAP: Interactive emotional dyadic motion capture database},
  volume = {42},
  year = {2008},
}
""",
        descriptive_stats={
            "n_samples": {"train": 10039},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "gender_id"
    samples_per_label: int = 100
    is_cross_validation: bool = True

    def dataset_transform(self):
        # Define label mapping
        label2id = {"Female": 0, "Male": 1}

        # Apply transformation to all dataset splits
        for split in self.dataset:
            # Define transform function to add numeric labels
            def add_gender_id(example):
                example["gender_id"] = label2id[example["gender"]]
                return example

            print(f"Converting gender labels to numeric IDs for split '{split}'...")
            self.dataset[split] = self.dataset[split].map(add_gender_id)
