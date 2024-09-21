from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroshotClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class EuroSATClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="EuroSATZeroShot",
        description="Classifying satellite images.",
        reference="https://ieeexplore.ieee.org/document/8736785",
        dataset={
            "path": "timm/eurosat-rgb",
            "revision": "b4e28552cd5f3932b6abc37eb20d3e84901ad728",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2019-01-01",
            "2019-03-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Scene recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@ARTICLE{8736785,
        author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
        journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
        title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification}, 
        year={2019},
        volume={12},
        number={7},
        pages={2217-2226},
        keywords={Satellites;Earth;Remote sensing;Machine learning;Spatial resolution;Feature extraction;Benchmark testing;Dataset;deep convolutional neural network;deep learning;earth observation;land cover classification;land use classification;machine learning;remote sensing;satellite image classification;satellite images},
        doi={10.1109/JSTARS.2019.2918242}}
        """,
        descriptive_stats={
            "n_samples": {"test": 5400},
            "avg_character_length": {"test": 431.4},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        labels = [
            "annual crop land",
            "forest land",
            "brushland or shrubland",
            "highway or road",
            "industrial land",
            "pasture land",
            "permanent crop land",
            "residential land",
            "river",
            "sea or lake",
        ]
        return [f"a centered satellite photo of {name}." for name in labels]
