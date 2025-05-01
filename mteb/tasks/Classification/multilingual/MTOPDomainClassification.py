from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "th": ["tha-Thai"],
}


class MTOPDomainClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="MTOPDomainClassification",
        dataset={
            "path": "mteb/mtop_domain",
            "revision": "d80d48c1eb48d3562165c59d59d0034df9fff0bf",
            "trust_remote_code": True,
        },
        description="MTOP: Multilingual Task-Oriented Semantic Parsing",
        reference="https://arxiv.org/pdf/2008.09335.pdf",
        category="s2s",
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
  abstract = {Scaling semantic parsing models for task-oriented dialog systems to new languages is often expensive and time-consuming due to the lack of available datasets. Available datasets suffer from several shortcomings: a) they contain few languages b) they contain small amounts of labeled examples per language c) they are based on the simple intent and slot detection paradigm for non-compositional queries. In this paper, we present a new multilingual dataset, called MTOP, comprising of 100k annotated utterances in 6 languages across 11 domains. We use this dataset and other publicly available datasets to conduct a comprehensive benchmarking study on using various state-of-the-art multilingual pre-trained models for task-oriented semantic parsing. We achieve an average improvement of +6.3 points on Slot F1 for the two existing multilingual datasets, over best results reported in their experiments. Furthermore, we demonstrate strong zero-shot performance using pre-trained models combined with automatic translation and alignment, and a proposed distant supervision method to reduce the noise in slot label projection.},
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
