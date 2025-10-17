from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BirdsnapClassification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5
    label_column_name: str = "common"

    metadata = TaskMetadata(
        name="Birdsnap",
        description="Classifying bird images from 500 species.",
        reference="https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html",
        dataset={
            "path": "isaacchung/birdsnap",
            "revision": "fd23015508be94f0b5b59d61630e4ea2536509e4",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Berg_2014_CVPR,
  author = {Berg, Thomas and Liu, Jiongxin and Woo Lee, Seung and Alexander, Michelle L. and Jacobs, David W. and Belhumeur, Peter N.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  title = {Birdsnap: Large-scale Fine-grained Visual Categorization of Birds},
  year = {2014},
}
""",
    )
