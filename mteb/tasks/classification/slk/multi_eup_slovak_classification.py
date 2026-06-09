"""Multi-EuP v2 Slovak classification tasks.

This module implements two classification tasks based on the Multi-EuP v2 corpus,
using native Slovak speeches from the European Parliament to predict:
- Political party affiliation
- Gender of speakers

Note: Uses only speeches originally delivered in Slovak (LANGUAGE=SK) with the full
speech text (TEXT field), not translations.
"""

from typing import ClassVar

import datasets

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{yang-etal-2024-language-bias,
  author = {Yang, Jinrui and Jiang, Fan and Baldwin, Timothy},
  booktitle = {Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024)},
  doi = {10.18653/v1/2024.mrl-1.23},
  pages = {280--292},
  publisher = {Association for Computational Linguistics},
  title = {Language Bias in Multilingual Information Retrieval: The Nature of the Beast and Mitigation Methods},
  url = {https://aclanthology.org/2024.mrl-1.23/},
  year = {2024},
}
"""


class _MultiEupSlovakMixin:
    """Shared transformation logic for Multi-EuP v2 Slovak classification tasks.

    This mixin provides dataset transformation functionality for loading and processing
    the Multi-EuP v2 corpus. It uses native Slovak speeches (LANGUAGE=SK) with full
    speech text (TEXT field) for classification tasks.

    Note:
        Custom load_data() is required because the Multi-EuP v2 CSV file has mixed-type
        columns (columns 35 and 60: Irish Gaelic title and question ID contain both
        numeric and string values). Standard dataset loading fails with
        "Failed to parse string as double" errors. We use keep_default_na=False to
        force all columns to be treated as strings, avoiding type inference issues.

    Attributes:
        target_column: The column name to use as the classification label.
    """

    target_column: ClassVar[str]

    def load_data(self):
        """Load the Multi-EuP v2 dataset from HuggingFace.

        Note:
            We cannot use standard MTEB dataset loading (datasets.load_dataset(path))
            because the CSV file has mixed-type columns that cause pandas parsing errors.
            Instead, we explicitly load as CSV with keep_default_na=False.
        """
        if self.data_loaded:
            return

        # Construct the data file URL from metadata
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        data_files = {
            "train": f"https://huggingface.co/datasets/{path}/resolve/{revision}/clean_all_with_did_qid.MEP.csv"
        }

        # Load the CSV dataset with keep_default_na=False to avoid type inference issues
        # This is necessary because columns 35 (title_GA) and 60 (qid_GA) have mixed types
        # (some values are numeric, some are strings), which causes standard loading to fail
        self.dataset = datasets.load_dataset(
            "csv",
            data_files=data_files,
            keep_default_na=False,  # Treat all values as strings, don't infer types
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """Transform the Multi-EuP v2 dataset for classification.

        Performs the following steps:
        1. Filters for native Slovak speeches (LANGUAGE=SK) with valid text and labels
        2. Renames columns to standard MTEB format (text, label)
        3. Strips whitespace from text and labels
        4. Removes unnecessary columns
        5. Encodes labels as integers
        6. Creates train/test split (80/20) with stratification

        Note: MTEB classification requires both train and test splits. The train split
        is used to train a logistic regression classifier, and test is for evaluation.
        """
        dataset = self.dataset["train"]
        target = self.target_column

        def _has_required_fields(example: dict[str, str]) -> bool:
            """Check if example is native Slovak with non-empty text and label."""
            is_slovak = example.get("LANGUAGE") == "SK"
            text = example.get("TEXT") or ""
            label = example.get(target) or ""
            return is_slovak and bool(text.strip() and label.strip())

        # Filter for native Slovak speeches only
        dataset = dataset.filter(_has_required_fields)

        # Rename to standard MTEB column names
        dataset = dataset.rename_columns({"TEXT": "text", target: "label"})

        # Strip whitespace from text and labels
        dataset = dataset.map(
            lambda example: {
                "text": example["text"].strip(),
                "label": example["label"].strip(),
            }
        )

        # Remove all columns except text and label
        columns_to_remove = [
            column for column in dataset.column_names if column not in {"text", "label"}
        ]
        dataset = dataset.remove_columns(columns_to_remove)

        # Encode labels as integers
        dataset = dataset.class_encode_column("label")

        # Create train/test split (80/20) with stratification
        # Train split is used for training logistic regression, test for evaluation
        dataset = dataset.train_test_split(
            test_size=0.2,
            seed=self.seed,
            stratify_by_column="label",
        )

        self.dataset = dataset


class MultiEupSlovakPartyClassification(_MultiEupSlovakMixin, AbsTaskClassification):
    target_column: ClassVar[str] = "PARTY"

    metadata = TaskMetadata(
        name="MultiEupSlovakPartyClassification",
        description="Multi-class classification task to predict the European Parliament political group from native Slovak speeches in the Multi-EuP v2 corpus. Uses only speeches originally delivered in Slovak.",
        reference="https://aclanthology.org/2024.mrl-1.23/",
        dataset={
            "path": "unimelb-nlp/MultiEup-v2",
            "revision": "382817850e097286b3fa9d874fb1b5128d0a430c",
            # Note: Dataset loading is handled by custom load_data() method in _MultiEupSlovakMixin
            # due to mixed-type columns in the CSV file (see mixin docstring for details)
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-13", "2024-04-25"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt="Given a European Parliament deputy utterance as query, find the deputy's political group",
    )


class MultiEupSlovakGenderClassification(_MultiEupSlovakMixin, AbsTaskClassification):
    target_column: ClassVar[str] = "gender"

    metadata = TaskMetadata(
        name="MultiEupSlovakGenderClassification",
        description="Binary classification task to predict the gender of Members of the European Parliament from native Slovak speeches in the Multi-EuP v2 corpus. Uses only speeches originally delivered in Slovak.",
        reference="https://aclanthology.org/2024.mrl-1.23/",
        dataset={
            "path": "unimelb-nlp/MultiEup-v2",
            "revision": "382817850e097286b3fa9d874fb1b5128d0a430c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-13", "2024-04-25"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt="Given a European Parliament deputy utterance as query, find if the deputy is male or female",
    )
