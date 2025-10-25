from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class LibriCount(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="LibriCount",
        description="Multiclass speaker count identification. Dataset contains audio recordings with between 0 to 10 speakers.",
        reference="https://huggingface.co/datasets/silky1708/LibriCount",
        dataset={
            "path": "mteb/libricount",
            "revision": "cc851c56e30dc5dde80c1823de96d52ca3cb2607",
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
        bibtex_citation=r"""
@inproceedings{Stoter_2018,
  author = {Stoter, Fabian-Robert and Chakrabarty, Soumitro and Edler, Bernd and Habets, Emanuel A. P.},
  booktitle = {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/icassp.2018.8462159},
  month = apr,
  pages = {436-440},
  publisher = {IEEE},
  title = {Classification vs. Regression in Supervised Learning for Single Channel Speaker Count Estimation},
  url = {http://dx.doi.org/10.1109/ICASSP.2018.8462159},
  year = {2018},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
