from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishProductSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishProductSentimentClassification",
        description="Turkish Product Review Dataset",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "asparius/Turkish-Product-Review",
            "revision": "ad861e463abda351ff65ca5ac0cc5985afe9eb99",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        date=("2013-01-01", "2013-08-11"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{Demirtas2013CrosslingualPD,
            title={Cross-lingual polarity detection with machine translation},
            author={Erkin Demirtas and Mykola Pechenizkiy},
            booktitle={wisdom},
            year={2013},
            url={https://api.semanticscholar.org/CorpusID:3912960}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 800},
            "avg_character_length": {"test": 246.85},
        },
    )
