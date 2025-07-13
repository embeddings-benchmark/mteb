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
            "path": "mteb/crema-d",
            "revision": "a9d0821955c418752248b8310438854e56a1cf34",
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
        bibtex_citation=r"""
@article{cao2014crema,
  author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal = {IEEE transactions on affective computing},
  number = {4},
  pages = {377--390},
  publisher = {IEEE},
  title = {Crema-d: Crowd-sourced emotional multimodal actors dataset},
  volume = {5},
  year = {2014},
}
""",
        descriptive_stats={
            "n_samples": {"train": 7442},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
