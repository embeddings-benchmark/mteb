from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 1024


class MovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MovieReviewSentimentClassification",
        dataset={
            "path": "allocine",
            "revision": "a4654f4896408912913a62ace89614879a549287",
        },
        description="The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.",
        reference="https://github.com/TheophileBlard/french-sentiment-analysis-with-bert",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2020-01-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@software{blard2020,
  title = {French sentiment analysis with BERT},
  author = {Théophile Blard},
  url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
  year = {2020},
}
""",
        n_samples={"validation": N_SAMPLES, "test": N_SAMPLES},
        avg_character_length={"validation": 550.3, "test": 558.1},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("review", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
