from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PoemSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PoemSentimentClassification",
        description="Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "poem_sentiment",
            "revision": "329d529d875a00c47ec71954a1a96ae167584770",
        },
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        text_creation="found",
        bibtex_citation="""
@misc{sheng2020investigating,
      title={Investigating Societal Biases in a Poetry Composition System},
      author={Emily Sheng and David Uthus},
      year={2020},
      eprint={2011.02686},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
        n_samples={"validation": 105, "test": 104},
        avg_character_length={"validation": 45.3, "test": 42.4},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("verse_text", "text")
