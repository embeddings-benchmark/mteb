from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class STL10ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="STL10ZeroShot",
        description="Classifying 96x96 images from 10 classes.",
        reference="https://cs.stanford.edu/~acoates/stl10/",
        dataset={
            "path": "tanganke/stl10",
            "revision": "49ae7f94508f7feae62baf836db284306eab0b0f",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2011-01-01",
            "2011-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@InProceedings{pmlr-v15-coates11a,
        title = 	 {An Analysis of Single-Layer Networks in Unsupervised Feature Learning},
        author = 	 {Coates, Adam and Ng, Andrew and Lee, Honglak},
        booktitle = 	 {Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics},
        pages = 	 {215--223},
        year = 	 {2011},
        editor = 	 {Gordon, Geoffrey and Dunson, David and Dudík, Miroslav},
        volume = 	 {15},
        series = 	 {Proceedings of Machine Learning Research},
        address = 	 {Fort Lauderdale, FL, USA},
        month = 	 {11--13 Apr},
        publisher =    {PMLR},
        pdf = 	 {http://proceedings.mlr.press/v15/coates11a/coates11a.pdf},
        url = 	 {https://proceedings.mlr.press/v15/coates11a.html},
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 8000},
            "avg_character_length": {"test": 431.4},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
