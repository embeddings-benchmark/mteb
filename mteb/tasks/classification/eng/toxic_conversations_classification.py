from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
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
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jigsaw-unintended-bias-in-toxicity-classification,
  author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
  publisher = {Kaggle},
  title = {Jigsaw Unintended Bias in Toxicity Classification},
  url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
  year = {2019},
}
""",
        prompt="Classify the given comments as either toxic or not toxic",
        superseded_by="ToxicConversationsClassification.v2",
    )

    samples_per_label = 16

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class ToxicConversationsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicConversationsClassification.v2",
        description="Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview",
        dataset={
            "path": "mteb/toxic_conversations",
            "revision": "7ae55309fbe51a11e13c24887ceed200153514e9",
        },
        type="Classification",
        category="t2c",
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
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jigsaw-unintended-bias-in-toxicity-classification,
  author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
  publisher = {Kaggle},
  title = {Jigsaw Unintended Bias in Toxicity Classification},
  url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
  year = {2019},
}
""",
        prompt="Classify the given comments as either toxic or not toxic",
        adapted_from=["ToxicConversationsClassification"],
    )

    samples_per_label = 16

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
