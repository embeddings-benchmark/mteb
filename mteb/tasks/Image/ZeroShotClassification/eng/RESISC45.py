from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class RESISC45ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="RESISC45ZeroShot",
        description="Remote Sensing Image Scene Classification by Northwestern Polytechnical University (NWPU).",
        reference="https://ieeexplore.ieee.org/abstract/document/7891544",
        dataset={
            "path": "timm/resisc45",
            "revision": "fe12fc5f1b7606543b0355eda392f1ddc54625c6",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2017-03-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@ARTICLE{7891544,
        author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
        journal={Proceedings of the IEEE},
        title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
        year={2017},
        volume={105},
        number={10},
        pages={1865-1883},
        keywords={Remote sensing;Benchmark testing;Spatial resolution;Social network services;Satellites;Image analysis;Machine learning;Unsupervised learning;Classification;Benchmark data set;deep learning;handcrafted features;remote sensing image;scene classification;unsupervised feature learning},
        doi={10.1109/JPROC.2017.2675998}}
        """,
        descriptive_stats={
            "n_samples": {"test": 6300},
            "avg_character_length": {"test": 256},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"satellite imagery of {name}."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
