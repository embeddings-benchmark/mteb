from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
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


class MultiHateClassification(MultilingualTask, AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="MultiHateClassification",
        dataset={
            "path": "mteb/multi-hatecheck",
            "revision": "8f95949846bb9e33c6aaf730ccfdb8fe6bcfb7a9",
        },
        description="""Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 11 languages.
                     """,
        reference="https://aclanthology.org/2022.woah-1.15/",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2020-11-23", "2022-02-28"),
        domains=["Constructed", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{rottger-etal-2021-hatecheck,
  abstract = {Detecting online hate is a difficult task that even state-of-the-art models struggle with. Typically, hate speech detection models are evaluated by measuring their performance on held-out test data using metrics such as accuracy and F1 score. However, this approach makes it difficult to identify specific model weak points. It also risks overestimating generalisable model performance due to increasingly well-evidenced systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, we introduce HateCheck, a suite of functional tests for hate speech detection models. We specify 29 model functionalities motivated by a review of previous research and a series of interviews with civil society stakeholders. We craft test cases for each functionality and validate their quality through a structured annotation process. To illustrate HateCheck{'}s utility, we test near-state-of-the-art transformer models as well as two popular commercial models, revealing critical model weaknesses.},
  address = {Online},
  author = {R{\"o}ttger, Paul  and
Vidgen, Bertie  and
Nguyen, Dong  and
Waseem, Zeerak  and
Margetts, Helen  and
Pierrehumbert, Janet},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.4},
  editor = {Zong, Chengqing  and
Xia, Fei  and
Li, Wenjie  and
Navigli, Roberto},
  month = aug,
  pages = {41--58},
  publisher = {Association for Computational Linguistics},
  title = {{H}ate{C}heck: Functional Tests for Hate Speech Detection Models},
  url = {https://aclanthology.org/2021.acl-long.4},
  year = {2021},
}

@inproceedings{rottger-etal-2022-multilingual,
  abstract = {Hate speech detection models are typically evaluated on held-out test sets. However, this risks painting an incomplete and potentially misleading picture of model performance because of increasingly well-documented systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, recent research has thus introduced functional tests for hate speech detection models. However, these tests currently only exist for English-language content, which means that they cannot support the development of more effective models in other languages spoken by billions across the world. To help address this issue, we introduce Multilingual HateCheck (MHC), a suite of functional tests for multilingual hate speech detection models. MHC covers 34 functionalities across ten languages, which is more languages than any other hate speech dataset. To illustrate MHC{'}s utility, we train and test a high-performing multilingual hate speech detection model, and reveal critical model weaknesses for monolingual and cross-lingual applications.},
  address = {Seattle, Washington (Hybrid)},
  author = {R{\"o}ttger, Paul  and
Seelawi, Haitham  and
Nozza, Debora  and
Talat, Zeerak  and
Vidgen, Bertie},
  booktitle = {Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH)},
  doi = {10.18653/v1/2022.woah-1.15},
  editor = {Narang, Kanika  and
Mostafazadeh Davani, Aida  and
Mathias, Lambert  and
Vidgen, Bertie  and
Talat, Zeerak},
  month = jul,
  pages = {154--169},
  publisher = {Association for Computational Linguistics},
  title = {Multilingual {H}ate{C}heck: Functional Tests for Multilingual Hate Speech Detection Models},
  url = {https://aclanthology.org/2022.woah-1.15},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        # for each language perform some transforms
        for lang in self.dataset.keys():
            _dataset = self.dataset[lang]
            _dataset = _dataset.rename_columns({"is_hateful": "label"})
            for label in ["label", "functionality"]:
                _dataset = _dataset.class_encode_column(label)
            _dataset = _dataset["test"].train_test_split(
                train_size=1000,
                test_size=1000,
                seed=self.seed,
                stratify_by_column="functionality",
            )  # balanced sampling across types of hate speech
            _dataset = _dataset.remove_columns(["functionality"])
            self.dataset[lang] = _dataset
