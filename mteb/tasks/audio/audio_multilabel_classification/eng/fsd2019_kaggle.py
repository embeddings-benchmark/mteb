from __future__ import annotations

import datasets

from mteb.abstasks.audio.abs_task_multilabel_classification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.task_metadata import TaskMetadata


class FSD2019KaggleMultilingualClassification(
    MultilingualTask, AbsTaskAudioMultilabelClassification
):
    metadata = TaskMetadata(
        name="FSD2019Kaggle",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/confit/fsdkaggle2019-parquet",  # "https://huggingface.co/datasets/CLAPv2/FSD50K",
        dataset={
            "path": "confit/fsdkaggle2019-parquet",
            "revision": "648a5925c8013e345ae5d36bdda220b1d4b07f24",
        },  # this is actually used to download the data
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs={"curated": ["eng-Latn"], "noisy": ["eng-Latn"]},
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-01-30",
        ),  # Estimated date when this dataset was committed, what should be the second tuple?
        domains=["Web"],  # obtained from Freesound - online collaborative platform
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{eduardo_fonseca_2020_3612637,
  author = {Eduardo Fonseca and
Manoj Plakal and
Frederic Font and
Daniel P. W. Ellis and
Xavier Serra},
  doi = {10.5281/zenodo.3612637},
  month = jan,
  publisher = {Zenodo},
  title = {FSDKaggle2019},
  url = {https://doi.org/10.5281/zenodo.3612637},
  version = {1.0},
  year = {2020},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "sound"
    samples_per_label: int = 8

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=lang, **self.metadata_dict["dataset"]
            )

        self.dataset_transform()
        self.data_loaded = True
