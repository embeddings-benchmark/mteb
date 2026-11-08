from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GunshotTriangulation(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GunshotTriangulation",
        description="Classifying a weapon based on its muzzle blast",
        reference="https://huggingface.co/datasets/anime-sh/GunshotTriangulationHEAR",
        dataset={
            "path": "mteb/GunshotTriangulationHear",
            "revision": "d5caa98a10a41dd8890a343a631fdbaacc747108",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
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
@article{raponi2021soundgunsdigitalforensics,
  author = {Raponi, Simone and Oligeri, Gabriele and Ali, Isra Mohamed},
  doi = {10.1007/s11042-022-12612-w},
  issn = {1573-7721},
  journal = {Multimedia Tools and Applications},
  month = {Apr},
  number = {21},
  pages = {30387–30412},
  publisher = {Springer Science and Business Media LLC},
  title = {Sound of guns: digital forensics of gun audio samples meets artificial intelligence},
  url = {https://doi.org/10.1007/s11042-022-12612-w},
  volume = {81},
  year = {2022},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True
