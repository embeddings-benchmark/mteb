from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Mantis(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MantisClassification",
        description="Mantis",
        dataset={
            "path": "DeepPavlov/Mantis",
            "revision": "b1584c11cc2fddf315c843add60b767210c54841",
        },
        reference=None,
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=None,
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{penha2019introducingmantisnovelmultidomain,
    title={Introducing MANtIS: a novel Multi-Domain Information Seeking Dialogues Dataset}, 
    author={Gustavo Penha and Alexandru Balan and Claudia Hauff},
    year={2019},
    eprint={1912.04639},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/1912.04639}, 
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["dialog"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['message']}\n"
                    else:
                        text += f"Assistant: {entry['message']}\n"
            row["text"] = text
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                )
                .rename_column("category", "label")
            )
