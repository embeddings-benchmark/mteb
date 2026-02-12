from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Imagenet1kClassification(AbsTaskClassification):
    input_column_name: str = "jpg"
    label_column_name: str = "cls"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="Imagenet1k",
        description="ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.",
        reference="https://ieeexplore.ieee.org/document/5206848",
        dataset={
            "path": "mteb/wds_imagenet1k",
            "revision": "b54c9af96641d8a2cba5d1144629c724de94ef76",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2012-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{deng2009imagenet,
  author = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  journal = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
  organization = {Ieee},
  pages = {248--255},
  title = {ImageNet: A large-scale hierarchical image database},
  year = {2009},
}
""",
    )
