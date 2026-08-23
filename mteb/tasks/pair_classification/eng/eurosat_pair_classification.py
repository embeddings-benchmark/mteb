from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EuroSATPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="EuroSATPairClassification",
        description="Classifying satellite image pairs as showing the same or different land use class, constructed from the EuroSAT RGB test split.",
        reference="https://ieeexplore.ieee.org/document/8736785",
        dataset={
            "path": "shriyasudhakar/EuroSATPairClassification",
            "revision": "2b7e45690555e1e984f67ffb9fdacd8c92490d4a",
        },
        type="ImagePairClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2019-01-01", "2019-03-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""@article{8736785,
  author = {Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  doi = {10.1109/JSTARS.2019.2918242},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  keywords = {Satellites;Earth;Remote sensing;Machine learning;Spatial resolution;Feature extraction;Benchmark testing;Dataset;deep convolutional neural network;deep learning;earth observation;land cover classification;land use classification;machine learning;remote sensing;satellite image classification;satellite images},
  number = {7},
  pages = {2217-2226},
  title = {EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  volume = {12},
  year = {2019},
}""",
    )

    input1_column_name: str = "image1"
    input2_column_name: str = "image2"
    label_column_name: str = "label"
