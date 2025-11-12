from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class YueOpenriceReviewClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YueOpenriceReviewClassification",
        description="A Cantonese dataset for review classification",
        reference="https://github.com/Christainx/Dataset_Cantonese_Openrice",
        dataset={
            "path": "izhx/yue-openrice-review",
            "revision": "1300d045cf983bac23faadf3aa12a619624769da",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["yue-Hant"],
        main_score="accuracy",
        date=("2019-01-01", "2019-05-01"),
        domains=["Reviews", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xiang2019sentiment,
  author = {Xiang, Rong and Jiao, Ying and Lu, Qin},
  booktitle = {Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
  organization = {KDD WISDOM},
  pages = {1--9},
  title = {Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
  year = {2019},
}
""",
        superseded_by="YueOpenriceReviewClassification.v2",
    )

    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class YueOpenriceReviewClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YueOpenriceReviewClassification.v2",
        description="A Cantonese dataset for review classification This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/Christainx/Dataset_Cantonese_Openrice",
        dataset={
            "path": "mteb/yue_openrice_review",
            "revision": "702b7ebe3b3ac712f1c31e87ab7171b1f1ca6b6b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["yue-Hant"],
        main_score="accuracy",
        date=("2019-01-01", "2019-05-01"),
        domains=["Reviews", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xiang2019sentiment,
  author = {Xiang, Rong and Jiao, Ying and Lu, Qin},
  booktitle = {Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
  organization = {KDD WISDOM},
  pages = {1--9},
  title = {Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
  year = {2019},
}
""",
        adapted_from=["YueOpenriceReviewClassification"],
    )

    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
