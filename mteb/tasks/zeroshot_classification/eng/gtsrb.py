from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GTSRBZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GTSRBZeroShot",
        description="The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset for traffic signs. It consists of dataset of more than 50,000 traffic sign images. The dataset comprises 43 classes with unbalanced class frequencies.",
        reference="https://benchmark.ini.rub.de/",
        dataset={
            "path": "mteb/GTSRBZeroShot",
            "revision": "0ac34f3215677b9e9ae8325f3ec63b072bf6304b",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2011-01-01",
            "2011-12-01",
        ),  # Estimated range for the collection of reviews
        task_subtypes=["Activity recognition"],
        domains=["Scene"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{6033395,
  author = {Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle = {The 2011 International Joint Conference on Neural Networks},
  doi = {10.1109/IJCNN.2011.6033395},
  keywords = {Humans;Training;Image color analysis;Benchmark testing;Lead;Histograms;Image resolution},
  number = {},
  pages = {1453-1460},
  title = {The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
  volume = {},
  year = {2011},
}
""",
    )
