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
            "revision": "e09b9dea248d579376684268cbedba28cd66b9b4",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
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
        descriptive_stats={
            "n_samples": {"test": 1851},
            "avg_character_length": {"test": 431.4},
        },
    )

    # Override default column name in the subclass
    label_column_name: str = "common"
