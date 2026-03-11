from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AbgCosQA(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AbgCosQA",
        description="AbgCosQA",
        dataset={
            "path": "DeepPavlov/coqa_abg",
            "revision": "4e5afdd361b1de400231ec806166e46d0f652297",
        },
        reference=None,
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="f1",
        date=None,
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{guo2021abg,
    title={Abg-coqa: Clarifying ambiguity in conversational question answering},
    author={Guo, Meiqi and Zhang, Mingda and Reddy, Siva and Alikhani, Malihe},
    booktitle={3rd Conference on Automated Knowledge Base Construction},
    year={2021}
}""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            full_text = row["story"] + " "
            for turn in row["history_turns"]:
                full_text += (
                    "User: " + turn["question"] + " Assistant: " + turn["answer"] + " "
                )
            full_text += (
                "User: "
                + row["target_turn"]["question"]
                + " Assistant: "
                + row["target_turn"]["answer"]
            )
            row["text"] = full_text
            row["label"] = row["ambiguity"] == "ambiguous"
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                    num_proc=num_proc,
                )
                .select_columns(["text", "label"])
            )
