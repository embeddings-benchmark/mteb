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
    "pol": ["pol-Latn"],
    "por": ["por-Latn"],
    "spa": ["spa-Latn"],
}


def _transform(dataset):
    dataset = dataset.rename_columns({"is_hateful": "label"})
    for label in ["label", "functionality"]:
        dataset = dataset.class_encode_column(label)
    dataset = dataset["test"].train_test_split(
        train_size=1000, test_size=1000, seed=42, stratify_by_column="functionality"
    )  # balanced sampling across types of hate speech
    dataset = dataset.remove_columns(["functionality"])
    return dataset


class MultiHateClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultiHateClassification",
        dataset={
            "path": "mteb/multi-hatecheck",
            "revision": "ef137ea2b7c719183f8f60edf536b50f56d1365b",
        },
        description="""Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 11 languages.
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
        @inproceedings{rottger-etal-2021-hatecheck,
            title = "{H}ate{C}heck: Functional Tests for Hate Speech Detection Models",
            author = {R{\"o}ttger, Paul  and
            Vidgen, Bertie  and
            Nguyen, Dong  and
            Waseem, Zeerak  and
            Margetts, Helen  and
            Pierrehumbert, Janet},
            editor = "Zong, Chengqing  and
            Xia, Fei  and
            Li, Wenjie  and
            Navigli, Roberto",
            booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
            month = aug,
            year = "2021",
            address = "Online",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.acl-long.4",
            doi = "10.18653/v1/2021.acl-long.4",
            pages = "41--58",
            abstract = "Detecting online hate is a difficult task that even state-of-the-art models struggle with. Typically, hate speech detection models are evaluated by measuring their performance on held-out test data using metrics such as accuracy and F1 score. However, this approach makes it difficult to identify specific model weak points. It also risks overestimating generalisable model performance due to increasingly well-evidenced systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, we introduce HateCheck, a suite of functional tests for hate speech detection models. We specify 29 model functionalities motivated by a review of previous research and a series of interviews with civil society stakeholders. We craft test cases for each functionality and validate their quality through a structured annotation process. To illustrate HateCheck{'}s utility, we test near-state-of-the-art transformer models as well as two popular commercial models, revealing critical model weaknesses.",
        }

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
        n_samples={"test": 10000},
        avg_character_length={"test": 45.9},
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            metadata = self.metadata_dict.get("dataset", None)
            dataset = datasets.load_dataset(name=lang, **metadata)
            self.dataset[lang] = _transform(dataset)
        self.dataset_transform()
        self.data_loaded = True
