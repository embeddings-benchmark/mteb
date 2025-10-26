from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class SpokeNEnglishClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SpokeNEnglish",
        description="Human Sound Classification Dataset.",
        reference="https://zenodo.org/records/10810044",
        dataset={
            "path": "mteb/SpokeN-100-English",
            "revision": "afbff14d927de14412d8124502313ea6d9d140e0",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-01-01"),
        domains=["Spoken"],
        task_subtypes=["Vocal Sound Classification"],
        license="cc-by-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{groh2024spoken100crosslingualbenchmarkingdataset,
  archiveprefix = {arXiv},
  author = {René Groh and Nina Goes and Andreas M. Kist},
  eprint = {2403.09753},
  primaryclass = {cs.SD},
  title = {SpokeN-100: A Cross-Lingual Benchmarking Dataset for The Classification of Spoken Numbers in Different Languages},
  url = {https://arxiv.org/abs/2403.09753},
  year = {2024},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 8
    is_cross_validation: bool = True
