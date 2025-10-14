from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SUN397Classification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="SUN397",
        description="Large scale scene recognition in 397 categories.",
        reference="https://ieeexplore.ieee.org/abstract/document/5539970",
        dataset={
            "path": "dpdl-benchmark/sun397",
            "revision": "7e6af6a2499ad708618be868e1471eac0aca1168",
        },
        type="ImageClassification",
        category="i2c",
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
        modalities=["image"],
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
