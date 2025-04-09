from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class CREMA_DClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="CREMA_DClustering",
        description="Emotion clustering task with audio data for 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.",
        reference="https://huggingface.co/datasets/silky1708/CREMA-D",
        dataset={
            "path": "silky1708/CREMA-D",
            "revision": "ab26a0ddbeade7c31a3208ecc043f06f9953892c",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2014-01-01", "2014-12-31"),
        domains=["Speech"],
        task_subtypes=["Emotion Clustering"],
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
    ) 