from __future__ import annotations

import logging

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking

logger = logging.getLogger(__name__)

_EVAL_SPLIT = "dev"
_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "yo": ["yor-Latn"],
    "zh": ["zho-Hans"],
}

_CITATION = """@article{10.1162/tacl_a_00595,
    author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1114-1131},
    year = {2023},
    month = {09},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00595},
}"""


class MIRACLReranking(AbsTaskReranking, MultilingualTask):
    metadata = TaskMetadata(
        name="MIRACLReranking",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        reference="https://project-miracl.github.io/",
        dataset={
            "path": "miracl/mmteb-miracl-reranking",
            "revision": "6d1962c527217f8927fca80f890f14f36b2802af",
            "trust_remote_code": True,
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-06-01", "2023-01-30"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
        descriptive_stats={
            "n_samples": {"dev": 44608},
            "avg_character_length": {"dev": 506.30},
        },
    )
