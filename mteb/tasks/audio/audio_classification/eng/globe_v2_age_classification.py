from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GlobeV2AgeClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GLOBEV2Age",
        description="Age classification from the GLOBE v2 dataset (sampled and enhanced from CommonVoice dataset)",
        reference="https://huggingface.co/datasets/MushanW/GLOBE_V2",
        dataset={
            "path": "mteb/globe-v2-age-mini",
            "revision": "b36e803c88037b09688f7c915d93b4cd654ba67e",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-13", "2025-01-13"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Age Classification"],
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
    label_column_name: str = "age"

    is_cross_validation: bool = False
