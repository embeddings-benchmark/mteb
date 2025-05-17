from __future__ import annotations

import os

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class PatchCamelyonZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="PatchCamelyonZeroShot",
        description="""Histopathology diagnosis classification dataset.""",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24",
        dataset={
            "path": "clip-benchmark/wds_vtab-pcam",
            "revision": "502695fe1a141108650e3c5b91c8b5e0ff84ed49",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2018-01-01",
            "2018-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Medical"],
        task_subtypes=["Tumor detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{10.1007/978-3-030-00934-2_24,
  abstract = {We propose a new model for digital pathology segmentation, based on the observation that histopathology images are inherently symmetric under rotation and reflection. Utilizing recent findings on rotation equivariant CNNs, the proposed model leverages these symmetries in a principled manner. We present a visual analysis showing improved stability on predictions, and demonstrate that exploiting rotation equivariance significantly improves tumor detection performance on a challenging lymph node metastases dataset. We further present a novel derived dataset to enable principled comparison of machine learning models, in combination with an initial benchmark. Through this dataset, the task of histopathology diagnosis becomes accessible as a challenging benchmark for fundamental machine learning research.},
  address = {Cham},
  author = {Veeling, Bastiaan S.
and Linmans, Jasper
and Winkens, Jim
and Cohen, Taco
and Welling, Max},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2018},
  editor = {Frangi, Alejandro F.
and Schnabel, Julia A.
and Davatzikos, Christos
and Alberola-L{\'o}pez, Carlos
and Fichtinger, Gabor},
  isbn = {978-3-030-00934-2},
  pages = {210--218},
  publisher = {Springer International Publishing},
  title = {Rotation Equivariant CNNs for Digital Pathology},
  year = {2018},
}
""",
        descriptive_stats={
            "n_samples": {"test": 32768},
            "avg_character_length": {"test": 0},
        },
    )
    image_column_name = "webp"
    label_column_name = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, "templates/PatchCamelyon_labels.txt")) as f:
            labels = f.readlines()

        return [f"histopathology image of {c}" for c in labels]
