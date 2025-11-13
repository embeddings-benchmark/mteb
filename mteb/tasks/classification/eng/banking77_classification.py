from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
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
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{casanueva-etal-2020-efficient,
  address = {Online},
  author = {Casanueva, I{\~n}igo  and
Tem{\v{c}}inas, Tadas  and
Gerz, Daniela  and
Henderson, Matthew  and
Vuli{\'c}, Ivan},
  booktitle = {Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
  doi = {10.18653/v1/2020.nlp4convai-1.5},
  editor = {Wen, Tsung-Hsien  and
Celikyilmaz, Asli  and
Yu, Zhou  and
Papangelis, Alexandros  and
Eric, Mihail  and
Kumar, Anuj  and
Casanueva, I{\~n}igo  and
Shah, Rushin},
  month = jul,
  pages = {38--45},
  publisher = {Association for Computational Linguistics},
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  url = {https://aclanthology.org/2020.nlp4convai-1.5},
  year = {2020},
}
""",
        prompt="Given a online banking query, find the corresponding intents",
        superseded_by="Banking77Classification.v2",
    )


class Banking77ClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Banking77Classification.v2",
        description="Dataset composed of online banking queries annotated with their corresponding intents. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/2003.04807",
        dataset={
            "path": "mteb/banking77",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
        },
        type="Classification",
        category="t2c",
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
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{casanueva-etal-2020-efficient,
  address = {Online},
  author = {Casanueva, I{\~n}igo  and
Tem{\v{c}}inas, Tadas  and
Gerz, Daniela  and
Henderson, Matthew  and
Vuli{\'c}, Ivan},
  booktitle = {Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
  doi = {10.18653/v1/2020.nlp4convai-1.5},
  editor = {Wen, Tsung-Hsien  and
Celikyilmaz, Asli  and
Yu, Zhou  and
Papangelis, Alexandros  and
Eric, Mihail  and
Kumar, Anuj  and
Casanueva, I{\~n}igo  and
Shah, Rushin},
  month = jul,
  pages = {38--45},
  publisher = {Association for Computational Linguistics},
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  url = {https://aclanthology.org/2020.nlp4convai-1.5},
  year = {2020},
}
""",
        prompt="Given a online banking query, find the corresponding intents",
        adapted_from=["Banking77Classification"],
    )
