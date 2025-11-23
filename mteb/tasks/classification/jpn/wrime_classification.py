from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class WRIMEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WRIMEClassification",
        dataset={
            "path": "mteb/WRIMEClassification",
            "revision": "78cfd586d70d2753fe7080a29dfbc5c278b1d54d",
        },
        description="A dataset of Japanese social network rated for sentiment",
        reference="https://aclanthology.org/2021.naacl-main.169/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="accuracy",
        date=("2011-06-01", "2020-05-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="https://huggingface.co/datasets/shunk031/wrime#licensing-information",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kajiwara-etal-2021-wrime,
  address = {Online},
  author = {Kajiwara, Tomoyuki  and
Chu, Chenhui  and
Takemura, Noriko  and
Nakashima, Yuta  and
Nagahara, Hajime},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.169},
  editor = {Toutanova, Kristina  and
Rumshisky, Anna  and
Zettlemoyer, Luke  and
Hakkani-Tur, Dilek  and
Beltagy, Iz  and
Bethard, Steven  and
Cotterell, Ryan  and
Chakraborty, Tanmoy  and
Zhou, Yichao},
  month = jun,
  pages = {2095--2104},
  publisher = {Association for Computational Linguistics},
  title = {{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations},
  url = {https://aclanthology.org/2021.naacl-main.169},
  year = {2021},
}
""",
        superseded_by="WRIMEClassification.v2",
    )


class WRIMEClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WRIMEClassification.v2",
        dataset={
            "path": "mteb/wrime",
            "revision": "6687c3bd031a0b144189958bad57db0b95a48dec",
            "name": "ver2",
        },
        description="A dataset of Japanese social network rated for sentiment This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2021.naacl-main.169/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="accuracy",
        date=("2011-06-01", "2020-05-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="https://huggingface.co/datasets/shunk031/wrime#licensing-information",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kajiwara-etal-2021-wrime,
  address = {Online},
  author = {Kajiwara, Tomoyuki  and
Chu, Chenhui  and
Takemura, Noriko  and
Nakashima, Yuta  and
Nagahara, Hajime},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.169},
  editor = {Toutanova, Kristina  and
Rumshisky, Anna  and
Zettlemoyer, Luke  and
Hakkani-Tur, Dilek  and
Beltagy, Iz  and
Bethard, Steven  and
Cotterell, Ryan  and
Chakraborty, Tanmoy  and
Zhou, Yichao},
  month = jun,
  pages = {2095--2104},
  publisher = {Association for Computational Linguistics},
  title = {{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations},
  url = {https://aclanthology.org/2021.naacl-main.169},
  year = {2021},
}
""",
        adapted_from=["WRIMEClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
