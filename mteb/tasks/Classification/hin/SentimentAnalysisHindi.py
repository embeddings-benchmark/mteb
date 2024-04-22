from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SentimentAnalysisHindi(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentAnalysisHindi",
        description="Hindi Sentiment Analysis Dataset",
        reference="https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi",
        dataset={
            "path": "OdiaGenAI/sentiment_analysis_hindi",
            "revision": "1beac1b941da76a9c51e3e5b39d230fde9a80983",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["hin-Deva"],
        main_score="f1",
        date=None,
        form=["written"],
        dialect=[],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"train": 2497},
        avg_character_length={"train": 81.29},
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train"].select(range(2048))

