from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakParlaSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakParlaSentClassification",
        description="Slovak parliamentary sentiment classification dataset from the ParlaSent corpus. Contains sentences from parliamentary debates with 3-level sentiment annotations.",
        reference="https://huggingface.co/datasets/classla/ParlaSent",
        dataset={
            "path": "classla/ParlaSent",
            "name": "SK",
            "revision": "0587c2b6499fbc68a7623439c2af2b24748968dc",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2018-01-01", "2018-12-31"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{antunovic2022parlasent,
  author = {Antunovi{\'c}, Matej and Bra{\v{z}}inskas, Rytis and {\v{Z}}agar, Bojan and Haddow, Barry and Birch, Alexandra and Ljube{\v{s}}i{\'c}, Nikola},
  journal = {arXiv preprint arXiv:2210.03068},
  title = {ParlaSent: A multilingual sentiment analysis dataset of parliamentary debates},
  year = {2022},
}
""",
        prompt="Classify the sentiment expressed in the given text as negative, neutral or positive",
    )

    def dataset_transform(self, **kwargs) -> None:
        """Transform the ParlaSent dataset for classification.

        Note: MTEB classification requires both train and test splits.
        The train split is used to train a logistic regression classifier,
        and the test split is for evaluation.
        """
        # Rename 'sentence' column to 'text' as expected by MTEB
        dataset = self.dataset["train"].rename_columns({"sentence": "text"})

        # Encode label column as ClassLabel for stratification
        dataset = dataset.class_encode_column("label")

        # Create train/test split (80/20) with stratification to avoid data leakage
        self.dataset = dataset.train_test_split(
            test_size=0.2,
            seed=self.seed,
            stratify_by_column="label",
        )
