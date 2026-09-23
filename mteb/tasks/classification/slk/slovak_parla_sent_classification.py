from typing import Any

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
@inproceedings{mochtak-etal-2024-parlasent,
  address = {Torino, Italia},
  author = {Mochtak, Michal  and
Rupnik, Peter  and
Ljube{\v{s}}i{\'c}, Nikola},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  editor = {Calzolari, Nicoletta  and
Kan, Min-Yen  and
Hoste, Veronique  and
Lenci, Alessandro  and
Sakti, Sakriani  and
Xue, Nianwen},
  month = may,
  pages = {16024--16036},
  publisher = {ELRA and ICCL},
  title = {The {P}arla{S}ent Multilingual Training Dataset for Sentiment Identification in Parliamentary Proceedings},
  url = {https://aclanthology.org/2024.lrec-main.1393/},
  year = {2024},
}
""",
        prompt="Classify the sentiment expressed in the given text as negative, neutral or positive",
    )

    def dataset_transform(self, **kwargs: Any) -> None:
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
