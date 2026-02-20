from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Caltech101Classification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="Caltech101",
        description="Classifying images of 101 widely varied objects.",
        reference="https://ieeexplore.ieee.org/document/1384978",
        dataset={
            "path": "mteb/Caltech101",
            "revision": "011e51e5fb01f0c820824734edb7a539ab8e6650",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2003-01-01",
            "2004-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{1384978,
  author = {Li Fei-Fei and Fergus, R. and Perona, P.},
  booktitle = {2004 Conference on Computer Vision and Pattern Recognition Workshop},
  doi = {10.1109/CVPR.2004.383},
  keywords = {Bayesian methods;Testing;Humans;Maximum likelihood estimation;Assembly;Shape;Machine vision;Image recognition;Parameter estimation;Image databases},
  number = {},
  pages = {178-178},
  title = {Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories},
  volume = {},
  year = {2004},
}
""",
    )
