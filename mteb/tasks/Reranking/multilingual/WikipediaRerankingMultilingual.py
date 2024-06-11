from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
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
        hf_hub_name="ellamind/wikipedia-2023-11-reranking-multilingual",
        dataset={
            "path": "ellamind/wikipedia-2023-11-reranking-multilingual",
            "revision": "6268b37d6f975f2a134791ba2f250a91d0bdfb4f",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="map",
        date=("2023-11-01", "2024-05-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="LM-generated",
        dialect=[],
        text_creation="LM-generated and verified",
        bibtex_citation="""@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}""",
        n_samples={
            "en": 1500,
            "de": 1500,
            "it": 1500,
            "pt": 1500,
            "nl": 1500,
            "cs": 1500,
            "ro": 1500,
            "bg": 1500,
            "sr": 1500,
            "fi": 1500,
            "da": 1500,
            "fa": 1500,
            "hi": 1500,
            "bn": 1500,
            "no": 1500,
            "sv": 1500,
        },
        avg_character_length={"test": 452},
    )
