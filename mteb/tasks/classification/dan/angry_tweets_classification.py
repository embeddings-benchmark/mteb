from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AngryTweetsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AngryTweetsClassification",
        dataset={
            "path": "DDSC/angry-tweets",
            "revision": "20b0e6081892e78179356fada741b7afa381443d",
        },
        description="A sentiment dataset with 3 classes (positive, negative, neutral) for Danish tweets",
        reference="https://aclanthology.org/2021.nodalida-main.53/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{pauli2021danlp,
  author = {Pauli, Amalie Brogaard and Barrett, Maria and Lacroix, Oph{\'e}lie and Hvingelby, Rasmus},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  pages = {460--466},
  title = {DaNLP: An open-source toolkit for Danish Natural Language Processing},
  year = {2021},
}
""",
        prompt="Classify Danish tweets by sentiment. (positive, negative, neutral).",
        superseded_by="AngryTweetsClassification.v2",
    )

    samples_per_label = 16


class AngryTweetsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AngryTweetsClassification.v2",
        dataset={
            "path": "mteb/angry_tweets",
            "revision": "b9475fb66a13befda4fa9871cd92343bb2c0eb77",
        },
        description="A sentiment dataset with 3 classes (positive, negative, neutral) for Danish tweets This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2021.nodalida-main.53/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{pauli2021danlp,
  author = {Pauli, Amalie Brogaard and Barrett, Maria and Lacroix, Oph{\'e}lie and Hvingelby, Rasmus},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  pages = {460--466},
  title = {DaNLP: An open-source toolkit for Danish Natural Language Processing},
  year = {2021},
}
""",
        prompt="Classify Danish tweets by sentiment. (positive, negative, neutral).",
        adapted_from=["AngryTweetsClassification"],
    )

    samples_per_label = 16
