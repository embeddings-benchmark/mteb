from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuNLUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuNLUIntentClassification",
        dataset={
            "path": "mteb/RuNLUIntentClassification",
            "revision": "424d0f767aaa5c411e3a529eec04658e5726a39e",
        },
        description=(
            "Contains natural language data for human-robot interaction in home domain which we collected and"
            " annotated for evaluating NLU Services/platforms."
        ),
        reference="https://doi.org/10.1007/978-981-15-9323-9_15",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "rus-eng": [
                "rus-Cyrl",
                "eng-Latn",
            ],
            "rus": [
                "rus-Cyrl",
            ],
        },
        main_score="accuracy",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{liu2019benchmarkingnaturallanguageunderstanding,
  author = {Liu, Xingkun and Eshghi, Arash and Swietojanski, Pawel and Rieser, Verena},
  booktitle = {Increasing Naturalness and Flexibility in Spoken Dialogue Interaction},
  doi = {10.1007/978-981-15-9323-9_15},
  isbn = {9789811593239},
  issn = {1876-1119},
  pages = {165–183},
  publisher = {Springer Singapore},
  title = {Benchmarking Natural Language Understanding Services for Building Conversational Agents},
  url = {https://doi.org/10.1007/978-981-15-9323-9_15},
  year = {2021},
}
""",
    )
