from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class LibriCount(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="LibriCount",
        description="Multiclass speaker count identification. Dataset contains audio recordings with between 0 to 10 speakers.",
        reference="https://huggingface.co/datasets/silky1708/LibriCount",
        dataset={
            "path": "silky1708/LibriCount",
            "revision": "679b233f0ef8aa6eb58382efcda49d6557db1af7",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2017-01-01", "2017-12-31"),
        domains=["Speech"],
        task_subtypes=["Speaker Count Identification"],
        license="cc-by-4.0",
        annotations_creators="algorithmic",  # VAD (Voice Activity Detection) algo
        dialect=[],
        modalities=["audio"],
        sample_creation="created",  # from LibriSpeech dataset
        bibtex_citation="""@inproceedings{Stoter_2018,
            title={Classification vs. Regression in Supervised Learning for Single Channel Speaker Count Estimation},
            url={http://dx.doi.org/10.1109/ICASSP.2018.8462159},
            DOI={10.1109/icassp.2018.8462159},
            booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            publisher={IEEE},
            author={Stoter, Fabian-Robert and Chakrabarty, Soumitro and Edler, Bernd and Habets, Emanuel A. P.},
            year={2018},
            month=apr, pages={436-440}}
        """,
        descriptive_stats={
            "n_samples": {"train": 5720},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
