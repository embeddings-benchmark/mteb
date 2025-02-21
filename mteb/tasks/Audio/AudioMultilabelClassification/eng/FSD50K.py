from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSD50KClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="FSD50K",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/Fhrozen/FSD50k",
        dataset={
            "path": "Fhrozen/FSD50k",
            "revision": "67e4d8c2570caef0f90d48fdb756b337875d91db",
        },
        type="AudioMultilabelClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="mAP",
        date=(
            "2022-05-06",
            "2022-05-06",
        ),  # Estimated date when this dataset was committed, what should be the second tuple?
        domains=["Web"],  # obtained from Freesound - online collaborative platform
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=""" @misc{fonseca2022fsd50kopendatasethumanlabeled,
            title={FSD50K: An Open Dataset of Human-Labeled Sound Events}, 
            author={Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra},
            year={2022},
            eprint={2010.00475},
            archivePrefix={arXiv},
            primaryClass={cs.SD},
            url={https://arxiv.org/abs/2010.00475}, 
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 10231},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 8  # dunno what this is?
