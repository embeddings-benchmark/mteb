from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ToxicConversationsClassificationHumanSubset(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicConversationsClassificationHumanSubset",
        description="Human evaluation subset of Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.",
        reference="https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview",
        dataset={
            "path": "mteb/mteb-human-toxicity-classification",
            "revision": "3db547359a8e0d077f0c9df3f50e11ede67501b7",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2018-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jigsaw-unintended-bias-in-toxicity-classification,
  author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
  publisher = {Kaggle},
  title = {Jigsaw Unintended Bias in Toxicity Classification},
  url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
  year = {2019},
}
""",
        prompt="Classify the given comments as either toxic or not toxic",
    )

    samples_per_label = 16

    def load_data(self, **kwargs):
        """Load human test subset + full original training data"""
        # Load human evaluation subset
        human_dataset = load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
        )

        # Load full original training data
        original_dataset = load_dataset(
            "mteb/toxic_conversations_50k",
            revision="edfaf9da55d3dd50d43143d90c1ac476895ae6de",
        )

        # Use stratified subsampling to create train split from original test data
        # (since ToxicConversations only has test split in original)
        original_with_train = self.stratified_subsampling(
            original_dataset, seed=42, splits=["test"], n_samples=8000
        )

        # Combine: subsampled train + human test
        self.dataset = DatasetDict(
            {
                "train": original_with_train["test"],  # This becomes our training data
                "test": human_dataset["test"],
            }
        )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
