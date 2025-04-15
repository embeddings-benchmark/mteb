from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BirdCLEFClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="BirdCLEF",
        description="BirdCLEF+ 2025 dataset for species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia.",
        reference="https://huggingface.co/datasets/christopher/birdclef-2025",
        dataset={
            "path": "christopher/birdclef-2025",
            "revision": "dad4270214702fb42482c302d0710cbc820d95ef",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],  # Using train split as it contains the labeled data
        eval_langs=[
            "eng-Latn",
        ],
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),  # Competition year
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Classification"],
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
            "n_samples": {"train": 28000},  # Approximate number of rows
            "metadata": {
                "sampling_rate": 32000,
                "includes": ["birds", "amphibians", "mammals", "insects"],
                "region": "Middle Magdalena Valley of Colombia",
            },
        },
    )

    audio_column_name: str = "recording"
    label_column_name: str = "primary_label"
    samples_per_label: int = (
        50  # This might need adjustment based on actual dataset statistics
    )
    is_cross_validation: bool = True
