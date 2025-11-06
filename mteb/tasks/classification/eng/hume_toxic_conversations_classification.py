from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HUMEToxicConversationsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HUMEToxicConversationsClassification",
        description="Human evaluation subset of Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.",
        reference="https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview",
        dataset={
            "path": "mteb/HUMEToxicConversationsClassification",
            "revision": "4c128c30566ffc7b01c7c3a367da20f36fc08ef8",
        },
        type="Classification",
        category="t2t",
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
