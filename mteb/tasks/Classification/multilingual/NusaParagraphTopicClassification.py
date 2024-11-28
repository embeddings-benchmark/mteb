from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "btk": ["bbc-Latn"],
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


class NusaParagraphTopicClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="NusaParagraphTopicClassification",
        dataset={
            "path": "gentaiscool/nusaparagraph_topic",
            "revision": "abb43f8d5b9510b8724b48283aca26c4733eac5d",
        },
        description="NusaParagraphTopicClassification is a multi-class topic classification on 10 Indonesian languages.",
        reference="https://github.com/IndoNLP/nusa-writes",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        domains=["Non-fiction", "Fiction", "Written"],
        task_subtypes=["Topic classification"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{cahyawijaya-etal-2023-nusawrites,
            title = "NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages",
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
    )
