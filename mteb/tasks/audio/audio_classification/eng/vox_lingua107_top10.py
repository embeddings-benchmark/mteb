from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class VoxLingua107Top10(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxLingua107_Top10",
        description="Spoken Language Identification for a given audio samples (10 classes/languages)",
        reference="https://huggingface.co/datasets/silky1708/VoxLingua107-Top-10",
        dataset={
            "path": "mteb/voxlingua107-top10",
            "revision": "d934546d059e16c9a4669adbd518e0fa86a69ff0",
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
        bibtex_citation=r"""
@misc{valk2020voxlingua107datasetspokenlanguage,
  archiveprefix = {arXiv},
  author = {Jörgen Valk and Tanel Alumäe},
  eprint = {2011.12998},
  primaryclass = {eess.AS},
  title = {VoxLingua107: a Dataset for Spoken Language Recognition},
  url = {https://arxiv.org/abs/2011.12998},
  year = {2020},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
