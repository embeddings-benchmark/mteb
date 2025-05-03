from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterURLCorpus",
        dataset={
            "path": "mteb/twitterurlcorpus-pairclassification",
            "revision": "8b6510b0b1fa4e4c4f879467980e9be563ec1cdf",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
        category="s2s",
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
  abstract = {A major challenge in paraphrase research is the lack of parallel corpora. In this paper, we present a new method to collect large-scale sentential paraphrases from Twitter by linking tweets through shared URLs. The main advantage of our method is its simplicity, as it gets rid of the classifier or human in the loop needed to select data before annotation and subsequent application of paraphrase identification algorithms in the previous work. We present the largest human-labeled paraphrase corpus to date of 51,524 sentence pairs and the first cross-domain benchmarking for automatic paraphrase identification. In addition, we show that more than 30,000 new sentential paraphrases can be easily and continuously captured every month at {\textasciitilde}70{\%} precision, and demonstrate their utility for downstream NLP tasks through phrasal paraphrase extraction. We make our code and data freely available.},
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
