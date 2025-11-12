from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FinancialPhrasebankClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialPhrasebankClassification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://arxiv.org/abs/1307.5336",
        dataset={
            "path": "mteb/FinancialPhrasebankClassification",
            "revision": "44023d2ee430a3be2d2ee573801616d9e1392fbd",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2013-11-01", "2013-11-01"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Malo2014GoodDO,
  author = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal = {Journal of the Association for Information Science and Technology},
  title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
  volume = {65},
  year = {2014},
}
""",
        superseded_by="FinancialPhrasebankClassification.v2",
    )


class FinancialPhrasebankClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialPhrasebankClassification.v2",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/1307.5336",
        dataset={
            "path": "mteb/financial_phrasebank",
            "revision": "9349ecd31615a97081c245f5d7dbc0f4c6a1a656",
            "name": "sentences_allagree",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2013-11-01", "2013-11-01"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Malo2014GoodDO,
  author = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal = {Journal of the Association for Information Science and Technology},
  title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
  volume = {65},
  year = {2014},
}
""",
        adapted_from=["FinancialPhrasebankClassification"],
    )
