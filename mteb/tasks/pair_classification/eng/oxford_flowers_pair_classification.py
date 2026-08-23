from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class OxfordFlowersPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="OxfordFlowersPairClassification",
        description="Classifying image pairs as depicting the same or different flower species, constructed from the Oxford Flowers-102 test split.",
        reference="https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train",
        dataset={
            "path": "shriyasudhakar/OxfordFlowersPairClassification",
            "revision": "88e7a7e5cc08c7d9228f1c795313c564636f7eef",
        },
        type="ImagePairClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2012-01-01", "2015-12-31"),
        domains=["Reviews"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{4756141,
  author = {Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle = {2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing},
  doi = {10.1109/ICVGIP.2008.47},
  keywords = {Shape;Kernel;Distributed computing;Support vector machines;Support vector machine classification;object classification;segmentation},
  number = {},
  pages = {722-729},
  title = {Automated Flower Classification over a Large Number of Classes},
  volume = {},
  year = {2008},
}""",
    )

    input1_column_name: str = "image1"
    input2_column_name: str = "image2"
    label_column_name: str = "label"
