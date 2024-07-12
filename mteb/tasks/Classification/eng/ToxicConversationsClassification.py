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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2018-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY 4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{jigsaw-unintended-bias-in-toxicity-classification,
    author = {cjadams, Daniel Borkan, inversion, Jeffrey Sorensen, Lucas Dixon, Lucy Vasserman, nithum},
    title = {Jigsaw Unintended Bias in Toxicity Classification},
    publisher = {Kaggle},
    year = {2019},
    url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification}
}""",
        descriptive_stats={
            "n_samples": {"test": 50000},
            "avg_character_length": {"test": 296.6},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
