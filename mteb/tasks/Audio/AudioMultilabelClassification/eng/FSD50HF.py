from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSD50HFMultilingualClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="FSD50HF",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/Chand0320/fsd50k_hf",
        dataset={
            "path": "Chand0320/fsd50k_hf",
            "revision": "ca72d33100074e2933437e844028c941d8e8f065",
        },  # this is actually used to download the data
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
    label_column_name: str = "labels"
    samples_per_label: int = 8
