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
        bibtex_citation=r"""
@inproceedings{pmlr-v15-coates11a,
  address = {Fort Lauderdale, FL, USA},
  author = {Coates, Adam and Ng, Andrew and Lee, Honglak},
  booktitle = {Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics},
  editor = {Gordon, Geoffrey and Dunson, David and Dudík, Miroslav},
  month = {11--13 Apr},
  pages = {215--223},
  pdf = {http://proceedings.mlr.press/v15/coates11a/coates11a.pdf},
  publisher = {PMLR},
  series = {Proceedings of Machine Learning Research},
  title = {An Analysis of Single-Layer Networks in Unsupervised Feature Learning},
  url = {https://proceedings.mlr.press/v15/coates11a.html},
  volume = {15},
  year = {2011},
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
