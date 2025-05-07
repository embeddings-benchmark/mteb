from __future__ import annotations

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking

_EVAL_LANGS = {
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "nl": ["nld-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "sr": ["srp-Cyrl"],
    "no": ["nor-Latn"],
    "sv": ["swe-Latn"],
}


class WikipediaRerankingMultilingual(MultilingualTask, AbsTaskReranking):
    metadata = TaskMetadata(
        name="WikipediaRerankingMultilingual",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual",
        dataset={
            "path": "ellamind/wikipedia-2023-11-reranking-multilingual",
            "revision": "6268b37d6f975f2a134791ba2f250a91d0bdfb4f",
        },
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="map",
        date=("2023-11-01", "2024-05-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=r"""
@online{wikidump,
  author = {Wikimedia Foundation},
  title = {Wikimedia Downloads},
  url = {https://dumps.wikimedia.org},
}
""",
    )
