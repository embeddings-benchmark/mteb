from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class UrduRomanSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UrduRomanSentimentClassification",
        description="The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)",
        reference="https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set",
        dataset={
            "path": "roman_urdu",
            "revision": "566be6449bb30b9b9f2b59173391647fe0ca3224",
        },
        type="Classification",
        category="s2s",
        date=("2018-01-01", "2018-08-28"),
        eval_splits=["train"],
        eval_langs=["urd-Latn"],
        main_score="f1",
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{misc_roman_urdu_data_set_458,
  author       = {Sharf,Zareen},
  title        = {{Roman Urdu Data Set}},
  year         = {2018},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C58325}
}
    """,
        n_samples={"train": 20229},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "sentiment": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
