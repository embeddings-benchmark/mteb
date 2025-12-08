from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SwedishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwedishSentimentClassification",
        description="Dataset of Swedish reviews scarped from various public available websites",
        reference="https://huggingface.co/datasets/swedish_reviews",
        dataset={
            "path": "mteb/SwedishSentimentClassification",
            "revision": "39e35f55d58338ebd602f8d83b52cfe027f5146a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        superseded_by="SwedishSentimentClassification.v2",
    )


class SwedishSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwedishSentimentClassification.v2",
        description="Dataset of Swedish reviews scarped from various public available websites This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/swedish_reviews",
        dataset={
            "path": "mteb/swedish_sentiment",
            "revision": "f521560ac618eea57c85392c574c16b6c08c9487",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["SwedishSentimentClassification"],
    )
