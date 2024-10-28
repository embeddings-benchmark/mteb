from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ind-abs": ["ind-Latn", "abs-Latn"],
    "ind-btk": ["ind-Latn", "bbc-Latn"],
    "ind-bew": ["ind-Latn", "bew-Latn"],
    "ind-bhp": ["ind-Latn", "bhp-Latn"],
    "ind-jav": ["ind-Latn", "jav-Latn"],
    "ind-mad": ["ind-Latn", "mad-Latn"],
    "ind-mak": ["ind-Latn", "mak-Latn"],
    "ind-min": ["ind-Latn", "min-Latn"],
    "ind-mui": ["ind-Latn", "mui-Latn"],
    "ind-rej": ["ind-Latn", "rej-Latn"],
    "ind-sun": ["ind-Latn", "sun-Latn"],
}


class NusaTranslationBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="NusaTranslationBitextMining",
        dataset={
            "path": "gentaiscool/bitext_nusatranslation_miners",
            "revision": "ba52e9d114a4a145d79b4293afab31304a999a4c",
        },
        description="NusaTranslation is a parallel dataset for machine translation on 11 Indonesia languages and English.",
        reference="https://huggingface.co/datasets/indonlp/nusatranslation_mt",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        domains=["Social", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""
        @inproceedings{cahyawijaya2023nusawrites,
            title={NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
            author={Cahyawijaya, Samuel and Lovenia, Holy and Koto, Fajri and Adhista, Dea and Dave, Emmanuel and Oktavianti, Sarah and Akbar, Salsabil and Lee, Jhonson and Shadieq, Nuur and Cenggoro, Tjeng Wawan and others},
            booktitle={Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
            pages={921--945},
            year={2023}
        }

        """,
        descriptive_stats={
            "n_samples": {"train": 50200},
            "train": {
                "average_sentence1_length": 145.4552390438247,
                "average_sentence2_length": 148.56607569721115,
                "num_samples": 50200,
                "hf_subset_descriptive_stats": {
                    "ind-abs": {
                        "average_sentence1_length": 148.366,
                        "average_sentence2_length": 147.314,
                        "num_samples": 1000,
                    },
                    "ind-btk": {
                        "average_sentence1_length": 145.36666666666667,
                        "average_sentence2_length": 146.74045454545455,
                        "num_samples": 6600,
                    },
                    "ind-bew": {
                        "average_sentence1_length": 145.4280303030303,
                        "average_sentence2_length": 148.40530303030303,
                        "num_samples": 6600,
                    },
                    "ind-bhp": {
                        "average_sentence1_length": 133.528,
                        "average_sentence2_length": 128.138,
                        "num_samples": 1000,
                    },
                    "ind-jav": {
                        "average_sentence1_length": 145.42772727272728,
                        "average_sentence2_length": 145.8089393939394,
                        "num_samples": 6600,
                    },
                    "ind-mad": {
                        "average_sentence1_length": 145.35545454545453,
                        "average_sentence2_length": 153.6228787878788,
                        "num_samples": 6600,
                    },
                    "ind-mak": {
                        "average_sentence1_length": 145.42772727272728,
                        "average_sentence2_length": 150.6128787878788,
                        "num_samples": 6600,
                    },
                    "ind-min": {
                        "average_sentence1_length": 145.42772727272728,
                        "average_sentence2_length": 148.0621212121212,
                        "num_samples": 6600,
                    },
                    "ind-mui": {
                        "average_sentence1_length": 150.454,
                        "average_sentence2_length": 150.994,
                        "num_samples": 1000,
                    },
                    "ind-rej": {
                        "average_sentence1_length": 151.622,
                        "average_sentence2_length": 139.583,
                        "num_samples": 1000,
                    },
                    "ind-sun": {
                        "average_sentence1_length": 145.42772727272728,
                        "average_sentence2_length": 150.9880303030303,
                        "num_samples": 6600,
                    },
                },
            },
        },
    )
