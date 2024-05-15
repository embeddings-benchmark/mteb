from __future__ import annotations

from collections import Counter

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split

import numpy as np
from datasets import DatasetDict, Dataset

def multihot_encoder(labels):
    """Convert list of label lists into a 2-D multihot array"""
    label_set = set()
    for label_list in labels:
        label_set = label_set.union(set(label_list))
    label_set = sorted(label_set)

    multihot_vectors = []
    for label_list in labels:
        multihot_vectors.append([1 if x in label_list else 0 for x in label_set])

    return np.array(multihot_vectors)


def stratified_multilabel_subsampling(
    dataset_dict:DatasetDict,
    splits: list[str] = ["test"],
    label: str = "label",
    n_samples: int = 2048) -> DatasetDict:
    """Multilabel subsamples the dataset with stratification by the supplied label.
    Returns a datasetDict object.

    Args:
        dataset_dict: the DatasetDict object.
        splits: the splits of the dataset.
        label: the label with which the stratified sampling is based on.
        n_samples: Optional, number of samples to subsample. Default is max_n_samples.
    """

    for split in splits:
        n_split = len(dataset_dict[split])
        X_np = np.arange(n_split).reshape((-1, 1))
        labels_np = multihot_encoder(dataset_dict[split][label])
        _, _, test_idx, _ = iterative_train_test_split(
            X_np, labels_np,
            test_size=n_samples/n_split
        )
        dataset_dict.update({split: Dataset.from_dict(dataset_dict[split][test_idx])})
    return dataset_dict


class MalteseNewsClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="MalteseNewsClassification",
        description="""A multi-label topic classification dataset for Maltese News
        Articles. The data was collected from the press_mt subset from Korpus
        Malti v4.0. Article contents were cleaned to filter out JavaScript, CSS,
        & repeated non-Maltese sub-headings. The labels are based on the category
        field from this corpus. 
        """,
        reference="https://huggingface.co/datasets/MLRS/maltese_news_categories",
        dataset={
            "path": "MLRS/maltese_news_categories",
            "revision": "6bb0321659c4f07c4c2176c30c98c971be6571b4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["mlt-Latn"],
        main_score="accuracy",
        date=("2023-10-21", "2024-04-24"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{maltese-news-datasets,
            title = "Topic Classification and Headline Generation for {M}altese using a Public News Corpus",
            author = "Chaudhary, Amit Kumar  and
                    Micallef, Kurt  and
                    Borg, Claudia",
            booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
            month = may,
            year = "2024",
            publisher = "Association for Computational Linguistics",
        }""",
        n_samples={"train": 1146, "test": 1146},
        avg_character_length={"train": 1710.05, "test": 1797.47},
    )

    def dataset_transform(self):
        # 80% of categories have just one label, so it's safe to take the first
        self.dataset = self.dataset.rename_columns({"labels" :"label"})
        remove_cols = [
            col for col in self.dataset["test"].column_names if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(remove_cols)
        # split dataset
        self.dataset = stratified_multilabel_subsampling(
            self.dataset, splits=["train", "test"], n_samples=2048
        )
