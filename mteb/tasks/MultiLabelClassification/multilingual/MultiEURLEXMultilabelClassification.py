from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class MultiEURLEXMultilabelClassification(
    MultilingualTask, AbsTaskMultilabelClassification
):
    metadata = TaskMetadata(
        name="MultiEURLEXMultilabelClassification",
        dataset={
            "path": "mteb/eurlex-multilingual",
            "revision": "2aea5a6dc8fdcfeca41d0fb963c0a338930bde5c",
        },
        description="EU laws in 23 EU languages containing annotated labels for 21 EUROVOC concepts.",
        reference="https://huggingface.co/datasets/coastalcph/multi_eurlex",
        category="p2p",
        modalities=["text"],
        type="MultilabelClassification",
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
            "es": ["spa-Latn"],
            "pl": ["pol-Latn"],
            "ro": ["ron-Latn"],
            "nl": ["nld-Latn"],
            "el": ["ell-Grek"],
            "hu": ["hun-Latn"],
            "pt": ["por-Latn"],
            "cs": ["ces-Latn"],
            "sv": ["swe-Latn"],
            "bg": ["bul-Cyrl"],
            "da": ["dan-Latn"],
            "fi": ["fin-Latn"],
            "sk": ["slk-Latn"],
            "lt": ["lit-Latn"],
            "hr": ["hrv-Latn"],
            "sl": ["slv-Latn"],
            "et": ["est-Latn"],
            "lv": ["lav-Latn"],
            "mt": ["mlt-Latn"],
        },
        main_score="accuracy",
        date=("1958-01-01", "2016-01-01"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chalkidis-etal-2021-multieurlex,
  author = {Chalkidis, Ilias
and Fergadiotis, Manos
and Androutsopoulos, Ion},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods
in Natural Language Processing},
  location = {Punta Cana, Dominican Republic},
  publisher = {Association for Computational Linguistics},
  title = {MultiEURLEX -- A multi-lingual and multi-label legal document
classification dataset for zero-shot cross-lingual transfer},
  url = {https://arxiv.org/abs/2109.00904},
  year = {2021},
}
""",
    )
