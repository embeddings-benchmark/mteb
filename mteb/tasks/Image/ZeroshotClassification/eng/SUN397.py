from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroshotClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SUN397Classification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="SUN397ZeroShot",
        description="Large scale scene recognition in 397 categories.",
        reference="https://ieeexplore.ieee.org/abstract/document/5539970",
        dataset={
            "path": "dpdl-benchmark/sun397",
            "revision": "7e6af6a2499ad708618be868e1471eac0aca1168",
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
        task_subtypes=["Scene recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@INPROCEEDINGS{5539970,
        author={Xiao, Jianxiong and Hays, James and Ehinger, Krista A. and Oliva, Aude and Torralba, Antonio},
        booktitle={2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition}, 
        title={SUN database: Large-scale scene recognition from abbey to zoo}, 
        year={2010},
        volume={},
        number={},
        pages={3485-3492},
        doi={10.1109/CVPR.2010.5539970}}
        """,
        descriptive_stats={
            "n_samples": {"test": 21750},
            "avg_character_length": {"test": 256},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        """Convert labels as such:
        - /b/boat_deck -> boat deck
        - /c/church/outdoor -> church outdoor
        """
        labels = []
        for name in self.dataset["test"].features[self.label_column_name].names:
            name = " ".join(name.split("/")[2:]).replace("_", " ")
            labels.append(f"a photo of a {name}.")
        return labels
