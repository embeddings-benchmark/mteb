from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxLingua107Top10(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxLingua107_Top10",
        description="Spoken Language Identification for a given audio samples (10 classes/languages)",
        reference="https://huggingface.co/datasets/silky1708/VoxLingua107-Top-10",
        dataset={
            "path": "silky1708/VoxLingua107-Top-10",
            "revision": "0ca67257f8b1a9ef8d2a526d9f669ed5c26ed6e7",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Speech"],
        task_subtypes=["Spoken Language Identification"],
        license="cc-by-4.0",
        annotations_creators="automatic-and-reviewed",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",  # from youtube
        bibtex_citation="""@misc{valk2020voxlingua107datasetspokenlanguage,
            title={VoxLingua107: a Dataset for Spoken Language Recognition}, 
            author={Jörgen Valk and Tanel Alumäe},
            year={2020},
            eprint={2011.12998},
            archivePrefix={arXiv},
            primaryClass={eess.AS},
            url={https://arxiv.org/abs/2011.12998}, 
        }""",
        descriptive_stats={
            "n_samples": {"train": 972},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
