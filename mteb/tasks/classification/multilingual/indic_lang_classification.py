from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "asm_Beng": ["asm-Beng"],
    "brx_Deva": ["brx-Deva"],
    "ben_Beng": ["ben-Beng"],
    "doi_Deva": ["doi-Deva"],
    "gom_Deva": ["gom-Deva"],
    "guj_Gujr": ["guj-Gujr"],
    "hin_Deva": ["hin-Deva"],
    "kan_Knda": ["kan-Knda"],
    "kas_Arab": ["kas-Arab"],
    "kas_Deva": ["kas-Deva"],
    "mai_Deva": ["mai-Deva"],
    "mal_Mlym": ["mal-Mlym"],
    "mar_Deva": ["mar-Deva"],
    "mni_Beng": ["mni-Beng"],
    "mni_Mtei": ["mni-Mtei"],
    "npi_Deva": ["npi-Deva"],
    "ory_Orya": ["ory-Orya"],
    "pan_Guru": ["pan-Guru"],
    "san_Deva": ["san-Deva"],
    "sat_Olck": ["sat-Olck"],
    "snd_Arab": ["snd-Arab"],
    "tam_Taml": ["tam-Taml"],
    "tel_Telu": ["tel-Telu"],
    "urd_Arab": ["urd-Arab"],
}

LANG_MAP = {
    ("Assamese", "Bengali"): "asm_Beng",
    ("Bodo", "Devanagari"): "brx_Deva",
    ("Bangla", "Bengali"): "ben_Beng",
    ("Konkani", "Devanagari"): "gom_Deva",
    ("Gujarati", "Gujarati"): "guj_Gujr",
    ("Hindi", "Devanagari"): "hin_Deva",
    ("Kannada", "Kannada"): "kan_Knda",
    ("Maithili", "Devanagari"): "mai_Deva",
    ("Malayalam", "Malayalam"): "mal_Mlym",
    ("Marathi", "Devanagari"): "mar_Deva",
    ("Nepali", "Devanagari"): "npi_Deva",
    ("Oriya", "Oriya"): "ory_Orya",
    ("Punjabi", "Gurmukhi"): "pan_Guru",
    ("Sanskrit", "Devanagari"): "san_Deva",
    ("Sindhi", "Perso - Arabic"): "snd_Arab",
    ("Tamil", "Tamil"): "tam_Taml",
    ("Telugu", "Telugu"): "tel_Telu",
    ("Urdu", "Perso - Arabic"): "urd_Arab",
    ("Kashmiri", "Perso - Arabic"): "kas_Arab",
    ("Kashmiri", "Devanagari"): "kas_Deva",
    ("Manipuri", "Meetei - Mayek"): "mni_Mtei",
    ("Manipuri", "Bengali"): "mni_Beng",
    ("Dogri", "Devanagari"): "doi_Deva",
    ("Santali", "Ol - Chiki"): "sat_Olck",
}


class IndicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndicLangClassification",
        dataset={
            "path": "mteb/IndicLangClassification",
            "revision": "36d9c05b5a4ba276554abc4eaaae87666d1e9c61",
        },
        description="A language identification test set for native-script as well as Romanized text which spans 22 Indic languages.",
        reference="https://arxiv.org/abs/2305.15814",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=[l for langs in _LANGUAGES.values() for l in langs],
        main_score="accuracy",
        date=("2022-08-01", "2023-01-01"),
        domains=["Web", "Non-fiction", "Written"],
        task_subtypes=["Language identification"],
        license="cc0-1.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{madhani-etal-2023-bhasa,
  address = {Toronto, Canada},
  author = {Madhani, Yash  and
Khapra, Mitesh M.  and
Kunchukuttan, Anoop},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  doi = {10.18653/v1/2023.acl-short.71},
  editor = {Rogers, Anna  and
Boyd-Graber, Jordan  and
Okazaki, Naoaki},
  month = jul,
  pages = {816--826},
  publisher = {Association for Computational Linguistics},
  title = {Bhasa-Abhijnaanam: Native-script and romanized Language Identification for 22 {I}ndic languages},
  url = {https://aclanthology.org/2023.acl-short.71},
  year = {2023},
}
""",
    )
