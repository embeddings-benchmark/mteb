from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class BirdCLEFClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="BirdCLEF",
        description="BirdCLEF+ 2025 dataset for species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia. Downsampled to 50 classes with 20 samples each.",
        reference="https://huggingface.co/datasets/christopher/birdclef-2025",
        dataset={
            "path": "mteb/birdclef25-mini",
            "revision": "582215665b247604b555da7ff4e071f82d3617db",
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
        bibtex_citation=r"""
@dataset{birdclef2025,
  author = {Christopher},
  publisher = {Hugging Face},
  title = {BirdCLEF+ 2025},
  url = {https://huggingface.co/datasets/christopher/birdclef-2025},
  year = {2025},
}
""",
    )

    audio_column_name: str = "recording"
    label_column_name: str = "primary_label"
    samples_per_label: int = 20
    is_cross_validation: bool = True
