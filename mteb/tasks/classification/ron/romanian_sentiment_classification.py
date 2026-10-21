from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class RomanianSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianSentimentClassification",
        description="An Romanian dataset for sentiment classification.",
        reference="https://aclanthology.org/2020.findings-emnlp.387/",
        dataset={
            "path": "mteb/RomanianSentimentClassification",
            "revision": "0f8df3d483924afb9130020e9f36ef09558fc9a1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-09-18", "2020-09-18"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dumitrescu2020birth,
  address = {Online},
  author = {Dumitrescu, Stefan and Avram, Andrei-Marius and Pyysalo, Sampo},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2020},
  doi = {10.18653/v1/2020.findings-emnlp.387},
  editor = {Cohn, Trevor and He, Yulan and Liu, Yang},
  month = {nov},
  pages = {4324--4328},
  publisher = {Association for Computational Linguistics},
  title = {The birth of {R}omanian {BERT}},
  url = {https://aclanthology.org/2020.findings-emnlp.387/},
  year = {2020},
}
""",
        superseded_by="RomanianSentimentClassification.v2",
    )


class RomanianSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianSentimentClassification.v2",
        description="An Romanian dataset for sentiment classification. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2020.findings-emnlp.387/",
        dataset={
            "path": "mteb/romanian_sentiment",
            "revision": "bf545b83db13cf73ed402749b21a7777e0afdc6a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-09-18", "2020-09-18"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dumitrescu2020birth,
  address = {Online},
  author = {Dumitrescu, Stefan and Avram, Andrei-Marius and Pyysalo, Sampo},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2020},
  doi = {10.18653/v1/2020.findings-emnlp.387},
  editor = {Cohn, Trevor and He, Yulan and Liu, Yang},
  month = {nov},
  pages = {4324--4328},
  publisher = {Association for Computational Linguistics},
  title = {The birth of {R}omanian {BERT}},
  url = {https://aclanthology.org/2020.findings-emnlp.387/},
  year = {2020},
}
""",
        adapted_from=["RomanianSentimentClassification"],
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
