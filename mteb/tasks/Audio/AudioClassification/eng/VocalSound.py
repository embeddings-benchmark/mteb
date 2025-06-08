from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VocalSoundClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VocalSound",
        description="Human Vocal Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/lmms-lab/vocalsound",
        dataset={
            "path": "lmms-lab/vocalsound",
            "revision": "f7a3562aa7841fabebfecf9df435160c8d55cb0c",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-01-01"),
        domains=["Spoken"],
        task_subtypes=["Vocal Sound Classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{Gong_2022,
            title={Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
            url={http://dx.doi.org/10.1109/ICASSP43922.2022.9746828},
            DOI={10.1109/icassp43922.2022.9746828},
            booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            publisher={IEEE},
            author={Gong, Yuan and Yu, Jin and Glass, James},
            year={2022},
            month=may }
                }""",
        descriptive_stats={
            "n_samples": {"test": 3594},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "answer"
