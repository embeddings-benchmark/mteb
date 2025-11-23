from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class YahooAnswersTopicsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YahooAnswersTopicsClassification",
        description="Dataset composed of questions and answers from Yahoo Answers, categorized into topics.",
        reference="https://huggingface.co/datasets/yahoo_answers_topics",
        dataset={
            "path": "mteb/YahooAnswersTopicsClassification",
            "revision": "900312fb5123b383a5c974c15fc1fbb9c3cd22af",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        domains=["Web", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{NIPS2015_250cf8b5,
  author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
  pages = {},
  publisher = {Curran Associates, Inc.},
  title = {Character-level Convolutional Networks for Text Classification},
  url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
  volume = {28},
  year = {2015},
}
""",
        superseded_by="YahooAnswersTopicsClassification.v2",
    )

    samples_per_label = 32


class YahooAnswersTopicsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YahooAnswersTopicsClassification.v2",
        description="Dataset composed of questions and answers from Yahoo Answers, categorized into topics. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/yahoo_answers_topics",
        dataset={
            "path": "mteb/yahoo_answers_topics",
            "revision": "c4d89f9633025d50954ab98a4c2c2feb188f6279",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        domains=["Web", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{NIPS2015_250cf8b5,
  author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
  pages = {},
  publisher = {Curran Associates, Inc.},
  title = {Character-level Convolutional Networks for Text Classification},
  url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
  volume = {28},
  year = {2015},
}
""",
        adapted_from=["YahooAnswersTopicsClassification"],
    )

    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
