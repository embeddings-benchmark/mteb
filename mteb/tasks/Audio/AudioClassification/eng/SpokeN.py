from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpokeNEnglishClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SpokeNEnglish",
        description="Human Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/lmms-lab/vocalsound",
        dataset={
            "path": "Mina76/SpokeN-100-English",
            "revision": "afbff14d927de14412d8124502313ea6d9d140e0",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
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
            "n_samples": {"validation": 1860, "test": 3590},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "answer"
    samples_per_label: int = 32


    def dataset_transform(self):
        self.dataset["train"] = self.dataset["test"]
