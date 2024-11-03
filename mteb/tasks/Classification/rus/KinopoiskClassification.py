from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2004-07-01", "2012-12-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{blinov2013research,
        title={Research of lexical approach and machine learning methods for sentiment analysis},
        author={Blinov, PD and Klekovkina, Maria and Kotelnikov, Eugeny and Pestov, Oleg},
        journal={Computational Linguistics and Intellectual Technologies},
        volume={2},
        number={12},
        pages={48--58},
        year={2013}
        }""",
        prompt="Classify the sentiment expressed in the given movie review text",
    )
