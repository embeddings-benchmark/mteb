from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class CREMA_D(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="CREMA_D",
        description="Emotion classification of audio into one of 6 classes: Anger, Disgust, Fear, Happy, Neutral, Sad.",
        reference="https://huggingface.co/datasets/silky1708/CREMA-D",
        dataset={
            "path": "silky1708/CREMA-D",
            "revision": "ab26a0ddbeade7c31a3208ecc043f06f9953892c",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2014-01-01", "2014-12-31"),
        domains=["Speech"],
        task_subtypes=["Emotion classification"],
        license="http://opendatacommons.org/licenses/odbl/1.0/",  # Open Database License
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@article{cao2014crema,
            title={Crema-d: Crowd-sourced emotional multimodal actors dataset},
            author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
            journal={IEEE transactions on affective computing},
            volume={5},
            number={4},
            pages={377--390},
            year={2014},
            publisher={IEEE}
            }""",
        descriptive_stats={
            "n_samples": {"train": 7442},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
