from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsClassification",
        dataset={
            "path": "mteb/AmazonReviewsClassification",
            "revision": "6b5d328eaae8ef408dd7d775040245cf86f92e9d",
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://aclanthology.org/2020.emnlp-main.369/",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Jpan"],
            "zh": ["cmn-Hans"],
        },
        main_score="accuracy",
        date=("2015-11-01", "2019-11-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="https://docs.opendata.aws/amazon-reviews-ml/license.txt",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{keung2020multilingual,
  address = {Online},
  author = {Keung, Phillip and Lu, Yichao and Szarvas, Gy{\"o}rgy and Smith, Noah A.},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  doi = {10.18653/v1/2020.emnlp-main.369},
  editor = {Webber, Bonnie and Cohn, Trevor and He, Yulan and Liu, Yang},
  month = {nov},
  pages = {4563--4568},
  publisher = {Association for Computational Linguistics},
  title = {The Multilingual {A}mazon Reviews Corpus},
  url = {https://aclanthology.org/2020.emnlp-main.369/},
  year = {2020},
}
""",
        prompt="Classify the given Amazon review into its appropriate rating category",
    )
