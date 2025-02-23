from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_classification import AbsTextClassification


class MovieReviewSentimentClassification(AbsTextClassification):
    metadata = TaskMetadata(
        name="MovieReviewSentimentClassification",
        dataset={
            "path": "tblard/allocine",
            "revision": "a4654f4896408912913a62ace89614879a549287",
        },
        description="The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.",
        reference="https://github.com/TheophileBlard/french-sentiment-analysis-with-bert",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@software{blard2020,
  title = {French sentiment analysis with BERT},
  author = {Théophile Blard},
  url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("review", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
