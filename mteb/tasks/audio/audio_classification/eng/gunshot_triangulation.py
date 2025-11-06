from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class GunshotTriangulation(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="GunshotTriangulation",
        description="Classifying a weapon based on its muzzle blast",
        reference="https://huggingface.co/datasets/anime-sh/GunshotTriangulationHEAR",
        dataset={
            "path": "mteb/GunshotTriangulationHear",
            "revision": "d5caa98a10a41dd8890a343a631fdbaacc747108",
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
        bibtex_citation=r"""
@misc{raponi2021soundgunsdigitalforensics,
  archiveprefix = {arXiv},
  author = {Simone Raponi and Isra Ali and Gabriele Oligeri},
  eprint = {2004.07948},
  primaryclass = {eess.AS},
  title = {Sound of Guns: Digital Forensics of Gun Audio Samples meets Artificial Intelligence},
  url = {https://arxiv.org/abs/2004.07948},
  year = {2021},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True
