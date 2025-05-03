from __future__ import annotations

import os

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTSRBZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="GTSRBZeroShot",
        description="""The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset for traffic signs. It consists of dataset of more than 50,000 traffic sign images. The dataset comprises 43 classes with unbalanced class frequencies.""",
        reference="https://benchmark.ini.rub.de/",
        dataset={
            "path": "clip-benchmark/wds_gtsrb",
            "revision": "1c13eff0803d2b02c9dc8dfe85e67770b3f0f3c5",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2011-01-01",
            "2011-12-01",
        ),  # Estimated range for the collection of reviews
        task_subtypes=["Activity recognition"],
        domains=["Scene"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{6033395,
  author = {Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle = {The 2011 International Joint Conference on Neural Networks},
  doi = {10.1109/IJCNN.2011.6033395},
  keywords = {Humans;Training;Image color analysis;Benchmark testing;Lead;Histograms;Image resolution},
  number = {},
  pages = {1453-1460},
  title = {The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
  volume = {},
  year = {2011},
}
""",
        descriptive_stats={
            "n_samples": {"test": 12630},
            "avg_character_length": {"test": 0},
        },
    )

    image_column_name: str = "webp"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, "templates/GTSRB_labels.txt")) as f:
            labels = f.readlines()

        return [f"a close up photo of a '{c}' traffic sign." for c in labels]
