from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BirdsnapClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="Birdsnap",
        description="Classifying bird images from 500 species.",
        reference="https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html",
        dataset={
            "path": "isaacchung/birdsnap",
            "revision": "fd23015508be94f0b5b59d61630e4ea2536509e4",
        },
        type="ImageClassification",
        category="i2i",
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
        bibtex_citation="""@InProceedings{Berg_2014_CVPR,
        author = {Berg, Thomas and Liu, Jiongxin and Woo Lee, Seung and Alexander, Michelle L. and Jacobs, David W. and Belhumeur, Peter N.},
        title = {Birdsnap: Large-scale Fine-grained Visual Categorization of Birds},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2014}
        }
        """,
    )

    # Override default column name in the subclass
    label_column_name: str = "common"
