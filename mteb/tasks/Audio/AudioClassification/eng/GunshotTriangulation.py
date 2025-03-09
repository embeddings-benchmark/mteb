from __future__ import annotations

from datasets import Audio

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class GunshotTriangulation(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="GunshotTriangulation",
        description="Classifying a weapon based on its muzzle blast",
        reference="https://huggingface.co/datasets/anime-sh/GunshotTriangulationHEAR",
        dataset={
            "path": "anime-sh/GunshotTriangulationHEAR",
            "revision": "cc57c7ff05daee3fcfd1657f18642167bf98e9e5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-03-07", "2025-03-07"),
        domains=[],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Gunshot Audio Classification"],
        license="not specified",  # Replace with appropriate license from allowed list
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@misc{raponi2021soundgunsdigitalforensics,
                title={Sound of Guns: Digital Forensics of Gun Audio Samples meets Artificial Intelligence}, 
                author={Simone Raponi and Isra Ali and Gabriele Oligeri},
                year={2021},
                eprint={2004.07948},
                archivePrefix={arXiv},
                primaryClass={eess.AS},
                url={https://arxiv.org/abs/2004.07948}, 
            }
        }""",
        descriptive_stats={
            "n_samples": {"train": 88},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    single_split_dataset: bool = True

    def dataset_transform(
        self,
    ):
        self.dataset = self.dataset.cast_column("audio", Audio())
