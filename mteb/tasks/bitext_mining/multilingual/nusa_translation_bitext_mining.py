from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

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


class NusaTranslationBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="NusaTranslationBitextMining",
        dataset={
            "path": "mteb/NusaTranslationBitextMining",
            "revision": "d08eca02c75a0b2ea9bfaaed1e42b7679588fa59",
        },
        description="NusaTranslation is a parallel dataset for machine translation on 11 Indonesia languages and English.",
        reference="https://huggingface.co/datasets/indonlp/nusatranslation_mt",
        type="BitextMining",
        category="t2t",
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
        bibtex_citation=r"""
@inproceedings{cahyawijaya-etal-2023-nusawrites,
  address = {Nusa Dua, Bali},
  author = {Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale},
  booktitle = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  editor = {Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa},
  month = nov,
  pages = {921--945},
  publisher = {Association for Computational Linguistics},
  title = {NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
  url = {https://aclanthology.org/2023.ijcnlp-main.60},
  year = {2023},
}
""",
    )
