from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WikipediaRerankingMultilingual(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WikipediaRerankingMultilingual",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual",
        hf_hub_name="ellamind/wikipedia-2023-11-reranking-multilingual",
        dataset={
            "path": "ellamind/wikipedia-2023-11-reranking-multilingual",
            "revision": "8f7f11b6fdb58296df57db1f0935c9697be88ef4",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs={
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
            "nl": ["dut-Latn"],
            "pt": ["por-Latn"],
            "ro": ["ron-Latn"],
            "sr": ["srp-Cyrl"],
        },
        main_score="map",
        date=("2023-11", "2024-05"),
        form="written",
        domains=["Encyclopaedic"],
        task_subtypes=["Reranking"],
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation=["created", "LM-generated and verified"],
        bibtex_citation=None,
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
        },
        avg_character_length=None,
    )
