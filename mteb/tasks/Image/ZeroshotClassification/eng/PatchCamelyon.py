from __future__ import annotations

import itertools
import os

from mteb.abstasks import AbsTaskZeroshotClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PatchCamelyonClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="PatchCamelyonZeroShot",
        description="""Histopathology diagnosis classification dataset.""",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24",
        dataset={
            "path": "clip-benchmark/wds_vtab-pcam",
            "revision": "502695fe1a141108650e3c5b91c8b5e0ff84ed49",
        },
        type="Classification",
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
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@InProceedings{10.1007/978-3-030-00934-2_24,
author="Veeling, Bastiaan S.
and Linmans, Jasper
and Winkens, Jim
and Cohen, Taco
and Welling, Max",
editor="Frangi, Alejandro F.
and Schnabel, Julia A.
and Davatzikos, Christos
and Alberola-L{\'o}pez, Carlos
and Fichtinger, Gabor",
title="Rotation Equivariant CNNs for Digital Pathology",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2018",
year="2018",
publisher="Springer International Publishing",
address="Cham",
pages="210--218",
abstract="We propose a new model for digital pathology segmentation, based on the observation that histopathology images are inherently symmetric under rotation and reflection. Utilizing recent findings on rotation equivariant CNNs, the proposed model leverages these symmetries in a principled manner. We present a visual analysis showing improved stability on predictions, and demonstrate that exploiting rotation equivariance significantly improves tumor detection performance on a challenging lymph node metastases dataset. We further present a novel derived dataset to enable principled comparison of machine learning models, in combination with an initial benchmark. Through this dataset, the task of histopathology diagnosis becomes accessible as a challenging benchmark for fundamental machine learning research.",
isbn="978-3-030-00934-2"
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