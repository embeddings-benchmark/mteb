from __future__ import annotations

import itertools

from mteb.abstasks import AbsTaskZeroshotClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTSRBZeroShotClassification(AbsTaskZeroshotClassification):
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
        domains=["Activity recognition"],
        task_subtypes=["Traffic sign recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
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

    image_column_name: str = "webp"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        labels = [
            "red and white circle 20 kph speed limit",
            "red and white circle 30 kph speed limit",
            "red and white circle 50 kph speed limit",
            "red and white circle 60 kph speed limit",
            "red and white circle 70 kph speed limit",
            "red and white circle 80 kph speed limit",
            "end / de-restriction of 80 kph speed limit",
            "red and white circle 100 kph speed limit",
            "red and white circle 120 kph speed limit",
            "red and white circle red car and black car no passing",
            "red and white circle red truck and black car no passing",
            "red and white triangle road intersection warning",
            "white and yellow diamond priority road",
            "red and white upside down triangle yield right-of-way",
            "stop",
            "empty red and white circle",
            "red and white circle no truck entry",
            "red circle with white horizonal stripe no entry",
            "red and white triangle with exclamation mark warning",
            "red and white triangle with black left curve approaching warning",
            "red and white triangle with black right curve approaching warning",
            "red and white triangle with black double curve approaching warning",
            "red and white triangle rough / bumpy road warning",
            "red and white triangle car skidding / slipping warning",
            "red and white triangle with merging / narrow lanes warning",
            "red and white triangle with person digging / construction / road work warning",
            "red and white triangle with traffic light approaching warning",
            "red and white triangle with person walking warning",
            "red and white triangle with child and person walking warning",
            "red and white triangle with bicyle warning",
            "red and white triangle with snowflake / ice warning",
            "red and white triangle with deer warning",
            "white circle with gray strike bar no speed limit",
            "blue circle with white right turn arrow mandatory",
            "blue circle with white left turn arrow mandatory",
            "blue circle with white forward arrow mandatory",
            "blue circle with white forward or right turn arrow mandatory",
            "blue circle with white forward or left turn arrow mandatory",
            "blue circle with white keep right arrow mandatory",
            "blue circle with white keep left arrow mandatory",
            "blue circle with white arrows indicating a traffic circle",
            "white circle with gray strike bar indicating no passing for cars has ended",
            "white circle with gray strike bar indicating no passing for trucks has ended",
        ]

        prompts = [
            [f"a zoomed in photo of a '{c}' traffic sign." for c in labels],
            [f"a centered photo of a '{c}'' traffic sign." for c in labels],
            [f"a close up photo of a '{c}'' traffic sign." for c in labels],
        ]

        return list(itertools.chain.from_iterable(prompts))
