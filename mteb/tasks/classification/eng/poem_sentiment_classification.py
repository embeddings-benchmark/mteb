from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PoemSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PoemSentimentClassification",
        description="Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "mteb/PoemSentimentClassification",
            "revision": "84af4753ebb04ca836fb54ce89a339839b03b748",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}
""",
        superseded_by="PoemSentimentClassification.v2",
    )


class PoemSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PoemSentimentClassification.v2",
        description="Poem Sentiment consist of poem verses from Project Gutenberg annotated for sentiment using the labels negative (0), positive (1), no_impact (2) and mixed (3). This version was corrected as a part of [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900) to fix common issues on the original datasets, including removing duplicates, and train-test leakage.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "mteb/poem_sentiment",
            "revision": "9fdc57b89ccc09a8d9256f376112d626878e51a7",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"), # a very rough guess of the date range of the poems, we do not have exact dates for all poems
        domains=["Written", "Fiction", "Poetry"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}
""",
        adapted_from=["PoemSentimentClassification"],
    )
