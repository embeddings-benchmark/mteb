from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class Banking77Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Banking77Classification",
        description="Dataset composed of online banking queries annotated with their corresponding intents.",
        reference="https://arxiv.org/abs/2003.04807",
        dataset={
            "path": "mteb/banking77",
            "revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2019-01-01",
            "2019-12-31",
        ),  # Estimated range for the collection of queries
        domains=["Written"],
        task_subtypes=[],
        license="MIT",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\~n}igo  and
      Tem{\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\'c}, Ivan",
    editor = "Wen, Tsung-Hsien  and
      Celikyilmaz, Asli  and
      Yu, Zhou  and
      Papangelis, Alexandros  and
      Eric, Mihail  and
      Kumar, Anuj  and
      Casanueva, I{\~n}igo  and
      Shah, Rushin",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45",
}""",
        descriptive_stats={
            "n_samples": {"test": 3080},
            "avg_character_length": {"test": 54.2},
        },
    )
