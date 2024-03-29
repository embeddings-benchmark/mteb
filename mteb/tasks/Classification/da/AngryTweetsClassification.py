from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AngryTweetsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AngryTweetsClassification",
        dataset={
            "path": "DDSC/angry-tweets",
            "revision": "20b0e6081892e78179356fada741b7afa381443d",
        },
        description="A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets",
        reference="https://aclanthology.org/2021.nodalida-main.53/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
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
        n_samples={"test": 1050},
        avg_character_length={"test": 156.1},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict
