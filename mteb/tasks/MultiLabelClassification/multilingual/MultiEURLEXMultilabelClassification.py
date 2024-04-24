from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskMultilabelClassification, MultilingualTask


class MultiEURLEXMultilabelClassification(
    MultilingualTask, AbsTaskMultilabelClassification
):
    metadata = TaskMetadata(
        name="MultiEURLEXMultilabelClassification",
        dataset={
            "path": "mteb/eurlex-multilingual",
            "revision": "2aea5a6dc8fdcfeca41d0fb963c0a338930bde5c",
        },
        description="EU laws in 23 EU languages containing gold labels.",
        reference="https://huggingface.co/datasets/coastalcph/multi_eurlex",
        category="p2p",
        type="MultilabelClassification",
        eval_splits=["validation", "test"],
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
        form=["written"],
        domains=["Legal", "Government"],
        task_subtypes=["Topic classification"],
        license="CC BY-SA 4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@InProceedings{chalkidis-etal-2021-multieurlex,
  author = {Chalkidis, Ilias  
                and Fergadiotis, Manos
                and Androutsopoulos, Ion},
  title = {MultiEURLEX -- A multi-lingual and multi-label legal document 
               classification dataset for zero-shot cross-lingual transfer},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2021},
  publisher = {Association for Computational Linguistics},
  location = {Punta Cana, Dominican Republic},
  url = {https://arxiv.org/abs/2109.00904}
}
        """,
        # TODO: count these
        n_samples={"validation": 0, "test": 0},
        avg_character_length={"validation": 0, "test": 0},
    )
