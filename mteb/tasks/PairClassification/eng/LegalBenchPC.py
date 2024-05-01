from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_DATASET_COLUMN_MAP = [
    {"name": "contract_qa", "sent1": "question", "sent2": "text", "labels": "answer"},
    {
        "name": "citation_prediction_classification",
        "sent1": "citation",
        "sent2": "text",
        "labels": "answer",
    },
    {
        "name": "consumer_contracts_qa",
        "sent1": "question",
        "sent2": "contract",
        "labels": "answer",
    },
]


class LegalBenchPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LegalBenchPC",
        description="LegalBench Dataset",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }""",
        n_samples={"test": 584},
        avg_character_length={"test": 74.75},
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        _hf_dataset = None
        for dataset_col_map in _DATASET_COLUMN_MAP:
            _dataset = datasets.load_dataset(
                self.metadata_dict["dataset"]["path"],
                dataset_col_map["name"],
                revision=self.metadata_dict["dataset"]["revision"],
                trust_remote_code=True,
            )

            _dataset = _dataset.rename_columns(
                {
                    dataset_col_map["sent1"]: "sent1",
                    dataset_col_map["sent2"]: "sent2",
                    dataset_col_map["labels"]: "labels",
                }
            )
            _dataset = _dataset.select_columns(["labels", "sent1", "sent2"])
            mapping = {"yes": 1, "no": 0}
            _dataset = _dataset.map(
                lambda example: {
                    "labels": mapping.get(example["labels"].lower(), example["labels"])
                }
            )

            if _hf_dataset is None:
                _hf_dataset = _dataset
            else:
                _hf_dataset["train"] = datasets.concatenate_datasets(
                    [_hf_dataset["train"], _dataset["train"]]
                )
                _hf_dataset["test"] = datasets.concatenate_datasets(
                    [_hf_dataset["test"], _dataset["test"]]
                )

        self.dataset = _hf_dataset
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sent1": hf_dataset["sent1"],
                    "sent2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
