from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

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


class MultiHateClassification(AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="MultiHateClassification",
        dataset={
            "path": "mteb/multi-hatecheck",
            "revision": "8f95949846bb9e33c6aaf730ccfdb8fe6bcfb7a9",
        },
        description="Hate speech detection dataset with binary (hateful vs non-hateful) labels. Includes 25+ distinct types of hate and challenging non-hate, and 11 languages.",
        reference="https://aclanthology.org/2022.woah-1.15/",
        type="Classification",
        category="t2c",
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
