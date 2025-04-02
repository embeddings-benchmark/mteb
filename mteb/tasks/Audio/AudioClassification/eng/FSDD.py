from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSDD(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="FSDD",
        description="Spoken digit classification of audio into one of 10 classes: 0-9",
        reference="https://huggingface.co/datasets/silky1708/Free-Spoken-Digit-Dataset",
        dataset={
            "path": "silky1708/Free-Spoken-Digit-Dataset",
            "revision": "f67112d5c77a50355e3232f58b1fa1917e62c435",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-10-06"),
        domains=["Music"],
        task_subtypes=["Spoken Digit Classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@misc{zohar2018free,
            author={J. Zohar and S. Cãar and F. Jason and P. Yuxin and N. Hereman and T. Adhish},
            title={Jakobovski/Free-Spoken-Digit-Dataset: V1.0.8},
            year={2018},
            month={aug},
            url={https://doi.org/10.5281/zenodo.1342401}
            }
        """,
        descriptive_stats={
            "n_samples": {"train": 2700, "test": 300},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
