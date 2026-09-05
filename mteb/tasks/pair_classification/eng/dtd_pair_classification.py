from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DTDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="DTDPairClassification",
        description="Classifying image pairs as depicting the same or different describable texture category, constructed from the DTD test split.",
        reference="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        dataset={
            "path": "shriyasudhakar/DTDPairClassification",
            "revision": "f819ae80cbf5c9c265df00e293bd765bec2383f7",
        },
        type="ImagePairClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-01-01", "2014-03-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{cimpoi14describing,
  author = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
  booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
  title = {Describing Textures in the Wild},
  year = {2014},
}""",
    )

    input1_column_name: str = "image1"
    input2_column_name: str = "image2"
    label_column_name: str = "label"
