from __future__ import annotations

from typing import Any

from mteb.abstasks.dialog_state_tracking import AbsTaskDST
from mteb.abstasks.task_metadata import TaskMetadata


class MultiWoz21Attraction(AbsTaskDST):
    n_experiments = 10
    classification_columns = (
        "attraction-area",
        "attraction-name",
        "attraction-type",
    )

    metadata = TaskMetadata(
        name="MultiWoz21Attraction",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"attraction": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultiWoz21Hospital(AbsTaskDST):
    n_experiments = 10
    classification_columns = ("hospital-department",)

    metadata = TaskMetadata(
        name="MultiWoz21Hospital",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"hospital": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultiWoz21Hotel(AbsTaskDST):
    n_experiments = 10
    classification_columns = (
        "hotel-area",
        "hotel-book day",
        "hotel-book people",
        "hotel-book stay",
        "hotel-internet",
        "hotel-name",
        "hotel-parking",
        "hotel-pricerange",
        "hotel-stars",
        "hotel-type",
    )

    metadata = TaskMetadata(
        name="MultiWoz21Hotel",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"hotel": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultiWoz21Restaurant(AbsTaskDST):
    n_experiments = 10
    classification_columns = (
        "restaurant-area",
        "restaurant-book day",
        "restaurant-book people",
        "restaurant-book time",
        "restaurant-food",
        "restaurant-name",
        "restaurant-pricerange",
    )

    metadata = TaskMetadata(
        name="MultiWoz21Restaurant",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"restaurant": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultiWoz21Taxi(AbsTaskDST):
    n_experiments = 10
    classification_columns = (
        "taxi-arriveby",
        "taxi-departure",
        "taxi-destination",
        "taxi-leaveat",
    )

    metadata = TaskMetadata(
        name="MultiWoz21Taxi",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"taxi": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultiWoz21Train(AbsTaskDST):
    n_experiments = 10
    classification_columns = (
        "train-arriveby",
        "train-book people",
        "train-day",
        "train-departure",
        "train-destination",
        "train-leaveat",
    )

    metadata = TaskMetadata(
        name="MultiWoz21Train",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"train": ["eng-Latn"]},
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )
