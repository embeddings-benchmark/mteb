from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ara": ["ara-Arab"],
    "cmn": ["cmn-Hans"],
    "eng": ["eng-Latn"],
    "deu": ["deu-Latn"],
    "fra": ["fra-Latn"],
    "hin": ["hin-Deva"],
    "ita": ["ita-Latn"],
    "nld": ["nld-Latn"],
    "por": ["por-Latn"],
    "spa": ["spa-Latn"],
}

_HF_AFFIX = {
    "ara": "arabic",
    "cmn": "mandarin",
    "eng": "",
    "deu": "german",
    "fra": "french",
    "hin": "hindi",
    "ita": "italian",
    "nld": "dutch",
    "por": "portuguese",
    "spa": "spanish",
}

_REVISION_DICT = {
    "ara": "65eb7455a05cb77b3ae0c69d444569a8eee54628",
    "cmn": "617d3e9fccd186277297cc305f6588af7384b008",
    "eng": "9d2ac89df04254e5c427bcc8d61b6d6c83a1f59b",
    "deu": "5229a5cc475f36c08d03ca52f0ccb005705e60d2",
    "fra": "5d3085f2129139abc10d2b58becd4d4f2978e5d5",
    "hin": "e9e68e1a4db04726b9278192377049d0f9693012",
    "ita": "21e3d5c827cb60619a89988b24979850a7af85a5",
    "nld": "d622427417d37a8d74e110e6289bc29af4ba4056",
    "por": "323bdf67e0fbd3d7f8086fad0971b5bd5a62524b",
    "spa": "a7ea759535bb9fad6361cca151cf94a46e88edf3",
}


def _transform(dataset):
    keep_cols = ["test_case", "label_gold"]
    rename_dict = dict(zip(keep_cols, ["text", "label"]))
    remove_cols = [col for col in dataset["test"].column_names if col not in keep_cols]
    dataset = dataset.rename_columns(rename_dict)
    dataset = dataset.class_encode_column("label")
    dataset = dataset.class_encode_column("functionality")
    dataset = dataset["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="functionality"
    )  # balanced sampling across types of hate speech
    dataset = dataset.remove_columns(remove_cols)
    return dataset


class MultiHateClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultiHateClassification",
        dataset={
            "path": "Paul/hatecheck",
            "revision": _REVISION_DICT,  # dynamic
        },
        description="""Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 10 languages.
                     """,
        reference="https://aclanthology.org/2022.woah-1.15/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2020-11-23", "2022-02-28"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @inproceedings{rottger-etal-2022-multilingual,
            title = "Multilingual {H}ate{C}heck: Functional Tests for Multilingual Hate Speech Detection Models",
            author = {R{\"o}ttger, Paul  and
            Seelawi, Haitham  and
            Nozza, Debora  and
            Talat, Zeerak  and
            Vidgen, Bertie},
            editor = "Narang, Kanika  and
            Mostafazadeh Davani, Aida  and
            Mathias, Lambert  and
            Vidgen, Bertie  and
            Talat, Zeerak",
            booktitle = "Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH)",
            month = jul,
            year = "2022",
            address = "Seattle, Washington (Hybrid)",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.woah-1.15",
            doi = "10.18653/v1/2022.woah-1.15",
            pages = "154--169",
            abstract = "Hate speech detection models are typically evaluated on held-out test sets. However, this risks painting an incomplete and potentially misleading picture of model performance because of increasingly well-documented systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, recent research has thus introduced functional tests for hate speech detection models. However, these tests currently only exist for English-language content, which means that they cannot support the development of more effective models in other languages spoken by billions across the world. To help address this issue, we introduce Multilingual HateCheck (MHC), a suite of functional tests for multilingual hate speech detection models. MHC covers 34 functionalities across ten languages, which is more languages than any other hate speech dataset. To illustrate MHC{'}s utility, we train and test a high-performing multilingual hate speech detection model, and reveal critical model weaknesses for monolingual and cross-lingual applications.",
        }
        """,
        n_samples={"test": 18250},
        avg_character_length={"test": 45.9},
    )

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            metadata = self.metadata_dict.get("dataset", None)
            path = f"{metadata['path']}-{_HF_AFFIX[lang]}".rstrip("-")
            dataset = datasets.load_dataset(
                path=path, revision=metadata["revision"][lang]
            )
            self.dataset[lang] = _transform(dataset)
        self.dataset_transform()
        self.data_loaded = True
