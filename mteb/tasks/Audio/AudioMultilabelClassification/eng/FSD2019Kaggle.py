from __future__ import annotations

import datasets

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks import MultilingualTask


class FSD2019KaggleClassification(AbsTaskAudioMultilabelClassification, MultilingualTask):
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
        bibtex_citation="""@dataset{eduardo_fonseca_2020_3612637,
                author       = {Eduardo Fonseca and
                                Manoj Plakal and
                                Frederic Font and
                                Daniel P. W. Ellis and
                                Xavier Serra},
                title        = {FSDKaggle2019},
                month        = jan,
                year         = 2020,
                publisher    = {Zenodo},
                version      = {1.0},
                doi          = {10.5281/zenodo.3612637},
                url          = {https://doi.org/10.5281/zenodo.3612637},
                }
        """,
        descriptive_stats={
            "n_samples": {"test": 8961},
        },
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
