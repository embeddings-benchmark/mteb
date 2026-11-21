from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SUN397ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SUN397ZeroShot",
        description="Large scale scene recognition in 397 categories.",
        reference="https://ieeexplore.ieee.org/abstract/document/5539970",
        dataset={
            "path": "mteb/SUN397ZeroShot",
            "revision": "18ee263f28a8948ea982ed282deec401dc544b73",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2017-03-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{5539970,
  author = {Xiao, Jianxiong and Hays, James and Ehinger, Krista A. and Oliva, Aude and Torralba, Antonio},
  booktitle = {2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2010.5539970},
  number = {},
  pages = {3485-3492},
  title = {SUN database: Large-scale scene recognition from abbey to zoo},
  volume = {},
  year = {2010},
}
""",
    )
