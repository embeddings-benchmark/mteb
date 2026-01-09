from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GlobeV2GenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GLOBEV2Gender",
        description="Gender classification from the GLOBE v2 dataset (sampled and enhanced from CommonVoice dataset)",
        reference="https://huggingface.co/datasets/MushanW/GLOBE_V2",
        dataset={
            "path": "diffunity/GLOBE_V2_test",
            "revision": "cc164ef46a8aa7ade377a2753260c3b9071d04eb",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-09", "2025-01-09"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{wang2024globe,
  archiveprefix = {arXiv},
  author = {Wenbin Wang and Yang Song and Sanjay Jha},
  eprint = {2406.14875},
  title = {GLOBE: A High-quality English Corpus with Global Accents for Zero-shot Speaker Adaptive Text-to-Speech},
  year = {2024},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "gender"

    is_cross_validation: bool = True
