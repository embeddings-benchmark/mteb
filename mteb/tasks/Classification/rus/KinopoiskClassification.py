from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class KinopoiskClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KinopoiskClassification",
        dataset={
            "path": "ai-forever/kinopoisk-sentiment-classification",
            "revision": "5911f26666ac11af46cb9c6849d0dc80a378af24",
        },
        description="Kinopoisk review sentiment classification",
        reference="https://www.dialog-21.ru/media/1226/blinovpd.pdf",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2004-07-01", "2012-12-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{blinov2013research,
        title={Research of lexical approach and machine learning methods for sentiment analysis},
        author={Blinov, PD and Klekovkina, Maria and Kotelnikov, Eugeny and Pestov, Oleg},
        journal={Computational Linguistics and Intellectual Technologies},
        volume={2},
        number={12},
        pages={48--58},
        year={2013}
        }""",
        n_samples={"validation": 1500, "test": 1500},
        avg_character_length={"validation": 1941.7, "test": 1897.3},
    )
