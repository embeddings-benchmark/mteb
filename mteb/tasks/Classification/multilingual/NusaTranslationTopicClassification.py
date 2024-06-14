from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = {
    "btk": ["btk-Latn"],
    "bew": ["bew-Latn"],
    "bug": ["bug-Latn"],
    "jav": ["jav-Latn"],
    "mad": ["mad-Latn"],
    "mak": ["mak-Latn"],
    "min": ["min-Latn"],
    "mui": ["mui-Latn"],
    "rej": ["rej-Latn"],
    "sun": ["sun-Latn"],
}


class NusaTranslationTopicClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="NusaTranslationTopicClassification",
        dataset={
            "path": "gentaiscool/nusatranslation_topic",
            "revision": "b53e037dabc0b2301bf328ca6c43cd23c030f779",
        },
        description="NusaTranslationTopicClassification is a multi-class topic classification on 10 Indonesian languages.",
        reference="https://github.com/IndoNLP/nusa-writes",
        category="s2s",
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=None,
        license="Apache 2.0",
        socioeconomic_status="mixed",
        annotations_creators="",
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{cahyawijaya-etal-2023-nusawrites,
    title = "{N}usa{W}rites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages",
    author = "Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale",
    editor = "Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.60",
    pages = "921--945",
}
""",
        n_samples={"train": 15516, "validation": 2948, "test": 6250},
        avg_character_length={"train": 5.51, "validation": 5.49, "test": 5.45},
    )
