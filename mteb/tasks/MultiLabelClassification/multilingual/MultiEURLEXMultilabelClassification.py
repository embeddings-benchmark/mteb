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
        description="EU laws in 23 EU languages containing gold labels.",
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
        license="CC BY-SA 4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
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
        descriptive_stats={
            "n_samples": {"test": 5000},
            "avg_character_length": {
                "test": {
                    "en": {
                        "average_text_length": 11720.2926,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "de": {
                        "average_text_length": 12865.4162,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "fr": {
                        "average_text_length": 13081.1098,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "it": {
                        "average_text_length": 12763.4786,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "es": {
                        "average_text_length": 13080.29,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "pl": {
                        "average_text_length": 12282.5926,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "ro": {
                        "average_text_length": 12836.9322,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "nl": {
                        "average_text_length": 12857.9742,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "el": {
                        "average_text_length": 12998.143,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "hu": {
                        "average_text_length": 12424.641,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "pt": {
                        "average_text_length": 12482.4616,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "cs": {
                        "average_text_length": 10783.4676,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "sv": {
                        "average_text_length": 11612.4774,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "bg": {
                        "average_text_length": 12235.4268,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "da": {
                        "average_text_length": 11773.958,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "fi": {
                        "average_text_length": 12087.6862,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "sk": {
                        "average_text_length": 11130.814,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "lt": {
                        "average_text_length": 11245.3566,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "hr": {
                        "average_text_length": 11022.142,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "sl": {
                        "average_text_length": 10620.0594,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "et": {
                        "average_text_length": 10898.4312,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "lv": {
                        "average_text_length": 10938.5102,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                    "mt": {
                        "average_text_length": 12589.7442,
                        "average_label_per_text": 3.5938,
                        "num_samples": 5000,
                        "unique_labels": 21,
                        "labels": {
                            "18": {"count": 2208},
                            "15": {"count": 1347},
                            "5": {"count": 1086},
                            "6": {"count": 1960},
                            "3": {"count": 2769},
                            "17": {"count": 1641},
                            "1": {"count": 653},
                            "20": {"count": 610},
                            "0": {"count": 774},
                            "2": {"count": 974},
                            "19": {"count": 444},
                            "9": {"count": 164},
                            "4": {"count": 394},
                            "10": {"count": 335},
                            "11": {"count": 531},
                            "7": {"count": 622},
                            "12": {"count": 513},
                            "8": {"count": 600},
                            "13": {"count": 102},
                            "14": {"count": 185},
                            "16": {"count": 57},
                        },
                    },
                }
            },
        },
    )
