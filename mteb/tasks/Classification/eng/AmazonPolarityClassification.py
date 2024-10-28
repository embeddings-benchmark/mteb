from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmazonPolarityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityClassification",
        description="Amazon Polarity Classification Dataset.",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "mteb/amazon_polarity",
            "revision": "e2d317d38cd51312af73b3d32a06d1a08b442046",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{McAuley2013HiddenFA,
  title={Hidden factors and hidden topics: understanding rating dimensions with review text},
  author={Julian McAuley and Jure Leskovec},
  journal={Proceedings of the 7th ACM conference on Recommender systems},
  year={2013},
  url={https://api.semanticscholar.org/CorpusID:6440341}
}""",
        descriptive_stats={
            "n_samples": {"test": 400000},
            "avg_character_length": {"test": 431.4},
        },
    )
