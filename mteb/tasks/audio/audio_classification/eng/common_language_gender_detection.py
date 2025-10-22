from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class CommonLanguageGenderDetection(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="CommonLanguageGenderDetection",
        description="Gender Classification. This is a stratified subsampled version of the original CommonLanguage datasets.",
        reference="https://huggingface.co/datasets/speechbrain/common_language",
        dataset={
            "path": "mteb/commonlanguage-gender-mini",
            "revision": "65cdf4a4565f09b1747cd8fb37d18cd9aa1f6dd9",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken", "Scene", "Speech"],
        task_subtypes=["Gender Classification", "Spoken Language Identification"],
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
    )

    audio_column_name: str = "audio"
    label_column_name: str = "gender"
    samples_per_label: int = 10

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name
        )
