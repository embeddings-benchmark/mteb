from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class YelpReviewFullClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YelpReviewFullClassification",
        description="Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "Yelp/yelp_review_full",
            "revision": "c1f9ee939b7d05667af864ee1cb066393154bf85",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2015-12-31"),  # reviews from 2015
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information",
        annotations_creators="derived",
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
        superseded_by="YelpReviewFullClassification.v2",
    )

    samples_per_label = 128

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class YelpReviewFullClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YelpReviewFullClassification.v2",
        description="Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "mteb/yelp_review_full",
            "revision": "49d71141934ae2e58733acd90908140e8ecaaee0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2015-12-31"),  # reviews from 2015
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information",
        annotations_creators="derived",
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
        adapted_from=["YelpReviewFullClassification"],
    )

    samples_per_label = 128

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
