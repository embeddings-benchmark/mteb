from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTSRBClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="GTSRB",
        description="""The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset for traffic signs. It consists of dataset of more than 50,000 traffic sign images. The dataset comprises 43 classes with unbalanced class frequencies.""",
        reference="https://benchmark.ini.rub.de/",
        dataset={
            "path": "clip-benchmark/wds_gtsrb",
            "revision": "1c13eff0803d2b02c9dc8dfe85e67770b3f0f3c5",
        },
        type="ImageClassification",
        category="i2i",
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
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@INPROCEEDINGS{6033395,
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle={The 2011 International Joint Conference on Neural Networks},
  title={The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
  year={2011},
  volume={},
  number={},
  pages={1453-1460},
  keywords={Humans;Training;Image color analysis;Benchmark testing;Lead;Histograms;Image resolution},
  doi={10.1109/IJCNN.2011.6033395}}
""",
        descriptive_stats={
            "n_samples": {"test": 12630},
            "avg_character_length": {"test": 0},
        },
    )
    image_column_name = "webp"
    label_column_name = "cls"
