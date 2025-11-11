from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DutchBookReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DutchBookReviewSentimentClassification",
        description="A Dutch book review for sentiment classification.",
        reference="https://github.com/benjaminvdb/DBRD",
        dataset={
            "path": "mteb/DutchBookReviewSentimentClassification",
            "revision": "1c2815ad38cf4794eb8d678fb08f569ea79392f6",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-1910-00896,
  archiveprefix = {arXiv},
  author = {Benjamin, van der Burgh and
Suzan, Verberne},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1910-00896.bib},
  eprint = {1910.00896},
  journal = {CoRR},
  timestamp = {Fri, 04 Oct 2019 12:28:06 +0200},
  title = {The merits of Universal Language Model Fine-tuning for Small Datasets
- a case with Dutch book reviews},
  url = {http://arxiv.org/abs/1910.00896},
  volume = {abs/1910.00896},
  year = {2019},
}
""",
        superseded_by="DutchBookReviewSentimentClassification.v2",
    )


class DutchBookReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DutchBookReviewSentimentClassification.v2",
        description="A Dutch book review for sentiment classification. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900). Additionally, a Dutch prompt was included.",
        reference="https://github.com/benjaminvdb/DBRD",
        dataset={
            "path": "mteb/dutch_book_review_sentiment",
            "revision": "73cffedb578b628588db03b8608880cf95688bb2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-1910-00896,
  archiveprefix = {arXiv},
  author = {Benjamin, van der Burgh and
Suzan, Verberne},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1910-00896.bib},
  eprint = {1910.00896},
  journal = {CoRR},
  timestamp = {Fri, 04 Oct 2019 12:28:06 +0200},
  title = {The merits of Universal Language Model Fine-tuning for Small Datasets
- a case with Dutch book reviews},
  url = {http://arxiv.org/abs/1910.00896},
  volume = {abs/1910.00896},
  year = {2019},
}
""",
        adapted_from=["DutchBookReviewSentimentClassification"],
        prompt={
            "query": "Classificeer de gegeven boekrecensie als positieve of negatieve sentiment"
        },
    )
