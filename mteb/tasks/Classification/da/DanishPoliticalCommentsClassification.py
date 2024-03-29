from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DanishPoliticalCommentsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DanishPoliticalCommentsClassification",
        dataset={
            "path": "danish_political_comments",
            "revision": "edbb03726c04a0efab14fc8c3b8b79e4d420e5a1",
        },
        description="A dataset of Danish political comments rated for sentiment",
        reference="https://huggingface.co/datasets/danish_political_comments",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["da"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"train": 9010},
        avg_character_length={"train": 69.9},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("target", "label")

        # create train and test splits
        self.dataset = self.dataset["train"].train_test_split(0.2, seed=self.seed)
