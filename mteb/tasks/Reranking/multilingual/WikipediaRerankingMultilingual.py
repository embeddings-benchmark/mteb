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
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="""@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}""",
        descriptive_stats={
            "n_samples": {
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
            "avg_character_length": {
                "test": {
                    "bg": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 60.82666666666667,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "bn": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 47.266666666666666,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "cs": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 56.272,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "da": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 56.75066666666667,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "de": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 70.004,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "en": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 68.372,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "fa": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 48.66733333333333,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "fi": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 55.343333333333334,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "hi": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 50.77733333333333,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "it": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 70.05466666666666,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "nl": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 65.34466666666667,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "pt": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 65.11933333333333,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "ro": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 61.973333333333336,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "sr": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 55.669333333333334,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "no": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 55.288,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                    "sv": {
                        "num_query": 1500,
                        "num_positive": 1500,
                        "num_negative": 1500,
                        "avg_query_len": 57.73,
                        "avg_positive_len": 1.0,
                        "avg_negative_len": 8.0,
                    },
                }
            },
        },
    )
