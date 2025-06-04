from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class UrduRomanSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UrduRomanSentimentClassification",
        description="The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)",
        reference="https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set",
        dataset={
            "path": "mteb/UrduRomanSentimentClassification",
            "revision": "905c1121c002c4b9adc4ebc5faaf4d6f50d1b1ee",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2018-01-01", "2018-08-28"),
        eval_splits=["train"],
        eval_langs=["urd-Latn"],
        main_score="f1",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{misc_roman_urdu_data_set_458,
  author = {Sharf,Zareen},
  howpublished = {UCI Machine Learning Repository},
  note = {{DOI}: https://doi.org/10.24432/C58325},
  title = {{Roman Urdu Data Set}},
  year = {2018},
}
""",
    )
