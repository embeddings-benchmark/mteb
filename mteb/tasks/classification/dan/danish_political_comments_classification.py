from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DanishPoliticalCommentsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DanishPoliticalCommentsClassification",
        dataset={
            "path": "mteb/DanishPoliticalCommentsClassification",
            "revision": "d743dcd5abb03d5ab357757a0e83522fc6696fcd",
        },
        description="A dataset of Danish political comments rated for sentiment",
        reference="https://huggingface.co/datasets/danish_political_comments",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=(
            "2000-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{SAMsentiment,
  author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
  institution = {IT University of Copenhagen},
  title = {Sentiment Analysis Multitool, SAM},
  year = {2019},
}
""",
        prompt="Classify Danish political comments for sentiment",
        superseded_by="DanishPoliticalCommentsClassification.v2",
    )

    samples_per_label = 16


class DanishPoliticalCommentsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DanishPoliticalCommentsClassification.v2",
        dataset={
            "path": "mteb/danish_political_comments",
            "revision": "476a9e7327aba70ad3e97a169d7310b86be9b245",
        },
        description="A dataset of Danish political comments rated for sentiment This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/danish_political_comments",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=(
            "2000-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{SAMsentiment,
  author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
  institution = {IT University of Copenhagen},
  title = {Sentiment Analysis Multitool, SAM},
  year = {2019},
}
""",
        prompt="Classify Danish political comments for sentiment",
        adapted_from=["DanishPoliticalCommentsClassification"],
    )

    samples_per_label = 16
