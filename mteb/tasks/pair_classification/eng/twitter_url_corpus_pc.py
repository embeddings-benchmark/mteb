from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TwitterURLCorpus(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterURLCorpus",
        dataset={
            "path": "mteb/twitterurlcorpus-pairclassification",
            "revision": "8b6510b0b1fa4e4c4f879467980e9be563ec1cdf",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{lan-etal-2017-continuously,
  address = {Copenhagen, Denmark},
  author = {Lan, Wuwei  and
Qiu, Siyu  and
He, Hua  and
Xu, Wei},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D17-1126},
  editor = {Palmer, Martha  and
Hwa, Rebecca  and
Riedel, Sebastian},
  month = sep,
  pages = {1224--1234},
  publisher = {Association for Computational Linguistics},
  title = {A Continuously Growing Dataset of Sentential Paraphrases},
  url = {https://aclanthology.org/D17-1126},
  year = {2017},
}
""",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
