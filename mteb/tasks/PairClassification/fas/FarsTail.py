from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FarsTail(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FarsTail",
        dataset={
            "path": "azarijafari/FarsTail",
            "revision": "7335288588f14e5a687d97fc979194c2abe6f4e7",
        },
        description="This dataset, named FarsTail, includes 10,367 samples which are provided in both the Persian language as well as the indexed format to be useful for non-Persian researchers. The samples are generated from 3,539 multiple-choice questions with the least amount of annotator interventions in a way similar to the SciTail dataset",
        reference="https://link.springer.com/article/10.1007/s00500-023-08959-3",
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ap",
        date=("2021-01-01", "2021-07-12"),  # best guess
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Textual Entailment"],
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{amirkhani2023farstail,
        title={FarsTail: a Persian natural language inference dataset},
        author={Amirkhani, Hossein and AzariJafari, Mohammad and Faridan-Jahromi, Soroush and Kouhkan, Zeinab and Pourjafari, Zohreh and Amirak, Azadeh},
        journal={Soft Computing},
        year={2023},
        publisher={Springer},
        doi={10.1007/s00500-023-08959-3}
        }""",
        n_samples={"test": 1029},  # after removing neutral
        avg_character_length={"test": 125.84},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        path = self.metadata_dict["dataset"]["path"]
        revision = self.metadata_dict["dataset"]["revision"]
        data_files = {
            "test": f"https://huggingface.co/datasets/{path}/resolve/{revision}/data/Test-word.csv"
        }
        self.dataset = datasets.load_dataset(
            "csv", data_files=data_files, delimiter="\t"
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        _dataset = {}
        self.dataset = self.dataset.filter(lambda x: x["label"] != "n")
        self.dataset = self.dataset.map(
            lambda example: {"label": 1 if example["label"] == "e" else 0}
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["premise"],
                    "sentence2": self.dataset[split]["hypothesis"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset
