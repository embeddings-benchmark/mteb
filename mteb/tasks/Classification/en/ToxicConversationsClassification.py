from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ToxicConversationsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicConversationsClassification",
        description="Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.",
        reference="https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview",
        dataset={
            "path": "mteb/toxic_conversations_50k",
            "revision": "edfaf9da55d3dd50d43143d90c1ac476895ae6de",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
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
        n_samples={"test": 50000},
        avg_character_length={"test": 296.6},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return dict(self.metadata)
