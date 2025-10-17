from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class OxfordFlowersClassification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="OxfordFlowersClassification",
        description="Classifying flowers",
        reference="https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train",
        dataset={
            "path": "nelorth/oxford-flowers",
            "revision": "a37b1891609c0376fa81eced756e7863e1bd873b",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{4756141,
  author = {Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle = {2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing},
  doi = {10.1109/ICVGIP.2008.47},
  keywords = {Shape;Kernel;Distributed computing;Support vector machines;Support vector machine classification;object classification;segmentation},
  number = {},
  pages = {722-729},
  title = {Automated Flower Classification over a Large Number of Classes},
  volume = {},
  year = {2008},
}
""",
    )
