from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

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
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        form=["written"],
        domains=["Social"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @inproceedings{cahyawijaya2023nusawrites,
            title={NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
            author={Cahyawijaya, Samuel and Lovenia, Holy and Koto, Fajri and Adhista, Dea and Dave, Emmanuel and Oktavianti, Sarah and Akbar, Salsabil and Lee, Jhonson and Shadieq, Nuur and Cenggoro, Tjeng Wawan and others},
            booktitle={Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
            pages={921--945},
            year={2023}
        }

        """,
        n_samples={"train": 50200},
        avg_character_length={"train": 147.01},
    )
