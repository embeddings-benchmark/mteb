from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RESISC45Classification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="RESISC45",
        description="Remote Sensing Image Scene Classification by Northwestern Polytechnical University (NWPU).",
        reference="https://ieeexplore.ieee.org/abstract/document/7891544",
        dataset={
            "path": "timm/resisc45",
            "revision": "fe12fc5f1b7606543b0355eda392f1ddc54625c6",
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
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{7891544,
  author = {Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
  doi = {10.1109/JPROC.2017.2675998},
  journal = {Proceedings of the IEEE},
  keywords = {Remote sensing;Benchmark testing;Spatial resolution;Social network services;Satellites;Image analysis;Machine learning;Unsupervised learning;Classification;Benchmark data set;deep learning;handcrafted features;remote sensing image;scene classification;unsupervised feature learning},
  number = {10},
  pages = {1865-1883},
  title = {Remote Sensing Image Scene Classification: Benchmark and State of the Art},
  volume = {105},
  year = {2017},
}
""",
    )
