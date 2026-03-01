from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GlobeV3GenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GLOBEV3Gender",
        description="Gender classification from the GLOBE v3 dataset (sampled and enhanced from CommonVoice dataset for TTS purpose). This dataset is a stratified and downsampled version of the original dataset, containing about 535 hours of speech data across 164 accents. We use the gender column as the target label for audio classification.",
        reference="https://huggingface.co/datasets/MushanW/GLOBE_V3",
        dataset={
            "path": "mteb/globe-v3-gender-mini",
            "revision": "7020a6c14ec8a8e967013e04f2a695ead308bee1",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-05-26", "2025-05-26"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
        license="cc0-1.0",
        annotations_creators="automatic",
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
    label_column_name: str = "predicted_gender"
