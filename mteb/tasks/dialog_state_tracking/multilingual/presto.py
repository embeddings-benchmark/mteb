from __future__ import annotations

from typing import Any

from datasets import load_dataset

import mteb
from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PrestoClassification(AbsTaskClassification):
    n_experiments = 1

    metadata = TaskMetadata(
        name="PrestoClassificationa",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/presto",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs=[
            "eng-Latn",
            "fra-Latn",
            "hin-Deva",
            "jpn-Jpan",
            "spa-Latn",
        ],
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

    def load_data(self, **kwargs):
        self.dataset = {}
        self.dataset["train"] = load_dataset(
            **self.metadata.dataset,
            name="train",
            split="train",
        )
        self.dataset["test"] = load_dataset(
            **self.metadata.dataset,
            name="test",
            split="train",
        )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["dialog"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            row["text"] = text
            row["dialog"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                    remove_columns=["dialog"],
                )
                .select_columns(["text", "label"])
            )


if __name__ == "__main__":
    model = mteb.get_model("minishlab/potion-base-2M")
    evaluator = mteb.MTEB([PrestoClassification()])

    evaluator.run(
        model,
        encode_kwargs={"batch_size": 32},
        co2_tracker=False,
    )
