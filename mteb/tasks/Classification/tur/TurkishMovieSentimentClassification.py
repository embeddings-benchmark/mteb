from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class TurkishMovieSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishMovieSentimentClassification",
        description="Turkish Movie Review Dataset",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "asparius/Turkish-Movie-Review",
            "revision": "409a4415cce5f6bcfca6d5f3ca3c408211ca00b3",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        date=None,
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""
        @inproceedings{Demirtas2013CrosslingualPD,
            title={Cross-lingual polarity detection with machine translation},
            author={Erkin Demirtas and Mykola Pechenizkiy},
            booktitle={wisdom},
            year={2013},
            url={https://api.semanticscholar.org/CorpusID:3912960}
        }
        """,
        n_samples={"train": 7972, "test": 2644},
        avg_character_length=None,
    )
