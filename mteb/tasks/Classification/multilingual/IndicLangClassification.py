from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

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
            "path": "ai4bharat/Bhasha-Abhijnaanam",
            "revision": "c54a95d9b9d62c891a03bd5da60715df7176b097",
        },
        description="A language identification test set for native-script as well as Romanized text which spans 22 Indic languages.",
        reference="https://arxiv.org/abs/2305.15814",
        category="s2s",
        type="Classification",
        eval_splits=["test"],
        eval_langs=[l for langs in _LANGUAGES.values() for l in langs],
        main_score="accuracy",
        date=("2022-08-01", "2023-01-01"),
        form=["written"],
        domains=["Web", "Non-fiction"],
        task_subtypes=["Language identification"],
        license="CC0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{madhani-etal-2023-bhasa,
    title = "Bhasa-Abhijnaanam: Native-script and romanized Language Identification for 22 {I}ndic languages",
    author = "Madhani, Yash  and
      Khapra, Mitesh M.  and
      Kunchukuttan, Anoop",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.71",
    doi = "10.18653/v1/2023.acl-short.71",
    pages = "816--826"
}""",
        n_samples={"test": 30418},
        avg_character_length={"test": 106.5},
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        labels = sorted(list(_LANGUAGES.keys()))

        data = datasets.load_dataset(**self.metadata_dict["dataset"])["train"]["data"][
            0
        ]

        dataset = {"train": [], "test": []}
        for lang, lang_code in LANG_MAP.items():
            subset = [
                item for item in data if (item["language"], item["script"]) == lang
            ]
            num_test_examples = min(2048, int(len(subset) * 0.7))
            subset = datasets.Dataset.from_list(subset).train_test_split(
                test_size=num_test_examples, seed=42
            )
            subset = subset.map(
                lambda x: {"lang_code": lang_code, "label": labels.index(lang_code)}
            )

            dataset["train"].append(subset["train"])
            dataset["test"].append(subset["test"])

        self.dataset = datasets.DatasetDict(
            {
                "train": datasets.concatenate_datasets(dataset["train"]),
                "test": datasets.concatenate_datasets(dataset["test"]),
            }
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.remove_columns(["language", "script"])
        self.dataset = self.dataset.rename_columns({"native sentence": "text"})
