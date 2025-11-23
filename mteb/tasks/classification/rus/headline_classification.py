from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HeadlineClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HeadlineClassification",
        dataset={
            "path": "ai-forever/headline-classification",
            "revision": "2fe05ee6b5832cda29f2ef7aaad7b7fe6a3609eb",
        },
        description="Headline rubric classification based on the paraphraser plus dataset.",
        reference="https://aclanthology.org/2020.ngt-1.6/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2009-01-01", "2020-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{gudkov-etal-2020-automatically,
  address = {Online},
  author = {Gudkov, Vadim  and
Mitrofanova, Olga  and
Filippskikh, Elizaveta},
  booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
  doi = {10.18653/v1/2020.ngt-1.6},
  editor = {Birch, Alexandra  and
Finch, Andrew  and
Hayashi, Hiroaki  and
Heafield, Kenneth  and
Junczys-Dowmunt, Marcin  and
Konstas, Ioannis  and
Li, Xian  and
Neubig, Graham  and
Oda, Yusuke},
  month = jul,
  pages = {54--59},
  publisher = {Association for Computational Linguistics},
  title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
  url = {https://aclanthology.org/2020.ngt-1.6},
  year = {2020},
}
""",
        prompt="Classify the topic or theme of the given news headline",
        superseded_by="HeadlineClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )


class HeadlineClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HeadlineClassification.v2",
        dataset={
            "path": "mteb/headline",
            "revision": "6bd88e7778ee2e3bd8d0ade1be3ad5b6d969145a",
        },
        description="Headline rubric classification based on the paraphraser plus dataset. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2020.ngt-1.6/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2009-01-01", "2020-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{gudkov-etal-2020-automatically,
  address = {Online},
  author = {Gudkov, Vadim  and
Mitrofanova, Olga  and
Filippskikh, Elizaveta},
  booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
  doi = {10.18653/v1/2020.ngt-1.6},
  editor = {Birch, Alexandra  and
Finch, Andrew  and
Hayashi, Hiroaki  and
Heafield, Kenneth  and
Junczys-Dowmunt, Marcin  and
Konstas, Ioannis  and
Li, Xian  and
Neubig, Graham  and
Oda, Yusuke},
  month = jul,
  pages = {54--59},
  publisher = {Association for Computational Linguistics},
  title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
  url = {https://aclanthology.org/2020.ngt-1.6},
  year = {2020},
}
""",
        prompt="Classify the topic or theme of the given news headline",
        adapted_from=["HeadlineClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
