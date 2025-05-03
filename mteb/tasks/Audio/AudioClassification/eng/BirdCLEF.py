from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BirdCLEFClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="BirdCLEF",
        description="BirdCLEF+ 2025 dataset for species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia. Downsampled to 50 classes with 20 samples each.",
        reference="https://huggingface.co/datasets/christopher/birdclef-2025",
        dataset={
            "path": "AdnanElAssadi/birdclef25_small",
            "revision": "55dbd1a0f77dd71980337a6e64620369c1e3585a",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=[
            "eng-Latn",
        ],
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),  # Competition year
        domains=["Spoken", "Speech", "Bioacoustics"],
        task_subtypes=["Species Classification"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@dataset{birdclef2025,
            author={Christopher},
            title={BirdCLEF+ 2025},
            year={2025},
            publisher={Hugging Face},
            url={https://huggingface.co/datasets/christopher/birdclef-2025}
        }""",
        descriptive_stats={
            "n_samples": {"train": 1000},  # 50 classes × 20 samples each
            "n_classes": 50,
        },
    )

    audio_column_name: str = "recording"
    label_column_name: str = "primary_label"
    samples_per_label: int = 20
    is_cross_validation: bool = True
