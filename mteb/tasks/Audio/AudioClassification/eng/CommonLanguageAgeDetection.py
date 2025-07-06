from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class CommonLanguageAgeDetection(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="CommonLanguageAgeDetection",
        description="Age Classification. This is a stratified subsampled version of the original CommonLanguage dataset.",
        reference="https://huggingface.co/datasets/speechbrain/common_language",
        dataset={
            "path": "mteb/commonlanguage-age-mini",
            "revision": "a9c585af68d65a29c4ad12121f83853fa1cdda92",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken", "Scene", "Speech"],
        task_subtypes=["Age Classification", "Spoken Language Identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{ganesh_sinisetty_2021_5036977,
  author = {Ganesh Sinisetty and
Pavlo Ruban and
Oleksandr Dymov and
Mirco Ravanelli},
  doi = {10.5281/zenodo.5036977},
  month = jun,
  publisher = {Zenodo},
  title = {CommonLanguage},
  url = {https://doi.org/10.5281/zenodo.5036977},
  version = {0.1},
  year = {2021},
}
""",
        descriptive_stats={
            "n_samples": {"train": 2000, "test": 2000, "validation": 2000}, 
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "age"
    samples_per_label: int = 10

    def dataset_transform(self):
        # remove rows where age is "not_defined" or "eighties" <- only 1 label so messes up stratified subsampling
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["age"] not in ["not_defined", "eighties"]
            )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name
        )
