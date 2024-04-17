from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DutchBookReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DutchBookReviewSentimentClassification",
        description="A Dutch book review for sentiment classification.",
        reference="https://github.com/benjaminvdb/DBRD",
        dataset={
            "path": "dbrd",
            "revision": "3f756ab4572e071eb53e887ab629f19fa747d39e",
        },
        type="Classification",
        category="s2s",
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-1910-00896,
  author    = {Benjamin van der Burgh and
               Suzan Verberne},
  title     = {The merits of Universal Language Model Fine-tuning for Small Datasets
               - a case with Dutch book reviews},
  journal   = {CoRR},
  volume    = {abs/1910.00896},
  year      = {2019},
  url       = {http://arxiv.org/abs/1910.00896},
  archivePrefix = {arXiv},
  eprint    = {1910.00896},
  timestamp = {Fri, 04 Oct 2019 12:28:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-00896.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
""",
        n_samples={"test": 2224},
        avg_character_length={"test": 1443.0},
    )
