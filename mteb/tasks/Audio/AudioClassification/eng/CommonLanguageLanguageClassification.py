from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class CommonLanguageLanguageClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="Common-Language-Language-Detection",
        description="Language Classification",
        reference="https://huggingface.co/datasets/speechbrain/common_language",
        dataset={
            "path": "speechbrain/common_language",
            "revision": "16ea653dd7d6a92f8fd80839466b1c6be1df300a",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken", "Scene", "Speech"],
        task_subtypes=["Language identification", "Spoken Language Identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@dataset{ganesh_sinisetty_2021_5036977,
  author       = {Ganesh Sinisetty and
                  Pavlo Ruban and
                  Oleksandr Dymov and
                  Mirco Ravanelli},
  title        = {CommonLanguage},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5036977},
  url          = {https://doi.org/10.5281/zenodo.5036977}
}
""",
        descriptive_stats={
            "n_samples": {"train": 22194, "test": 5963},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "language"
    samples_per_label: int = 10

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name
        )
