from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "th": ["tha-Thai"],
}


class MTOPDomainClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MTOPDomainClassification",
        dataset={
            "path": "mteb/MTOPDomainClassification",
            "revision": "a76d16fae880597b9c73047b50159220a441cb54",
        },
        description="MTOP: Multilingual Task-Oriented Semantic Parsing",
        reference="https://arxiv.org/pdf/2008.09335.pdf",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Spoken", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{li-etal-2021-mtop,
  address = {Online},
  author = {Li, Haoran  and
Arora, Abhinav  and
Chen, Shuohui  and
Gupta, Anchit  and
Gupta, Sonal  and
Mehdad, Yashar},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  doi = {10.18653/v1/2021.eacl-main.257},
  editor = {Merlo, Paola  and
Tiedemann, Jorg  and
Tsarfaty, Reut},
  month = apr,
  pages = {2950--2962},
  publisher = {Association for Computational Linguistics},
  title = {{MTOP}: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark},
  url = {https://aclanthology.org/2021.eacl-main.257},
  year = {2021},
}
""",
        prompt="Classify the intent domain of the given utterance in task-oriented conversation",
    )
