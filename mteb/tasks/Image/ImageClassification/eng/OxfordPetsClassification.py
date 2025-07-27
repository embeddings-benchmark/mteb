from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordPetsClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordPets",
        description="Classifying animal images.",
        reference="https://ieeexplore.ieee.org/abstract/document/6248092",
        dataset={
            "path": "isaacchung/OxfordPets",
            "revision": "557b480fae8d69247be74d9503b378a09425096f",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2009-01-01",
            "2010-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{6248092,
  author = {Parkhi, Omkar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, C. V.},
  booktitle = {2012 IEEE Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2012.6248092},
  keywords = {Positron emission tomography;Image segmentation;Cats;Dogs;Layout;Deformable models;Head},
  number = {},
  pages = {3498-3505},
  title = {Cats and dogs},
  volume = {},
  year = {2012},
}
""",
        descriptive_stats={
            "n_samples": {"test": 3669},
            "avg_character_length": {"test": 431.4},
        },
    )
