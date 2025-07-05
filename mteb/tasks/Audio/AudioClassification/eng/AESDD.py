from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AESDDClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="AESDD",
        description="Speech Emotion Recognition Dataset.",
        reference="https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/",
        dataset={
            "path": "EdwardLin2023/AESDD",
            "revision": "5ecde28811adf538a7699b76291a99503e3734f5",
            "trust_remote_code": True,
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2019-01-01"),
        domains=["Spoken"],
        task_subtypes=["Emotion recognition"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@article{vryzas2018speech,
            title={Speech emotion recognition for performance interaction},
            author={Vryzas, Nikolaos and Kotsakis, Rigas and Liatsou, Aikaterini and Dimoulas, Charalampos A and Kalliris, George},
            journal={Journal of the Audio Engineering Society},
            volume={66},
            number={6},
            pages={457--467},
            year={2018},
            publisher={Audio Engineering Society}
            }""",
        descriptive_stats={
            "n_samples": {"train": 604},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
