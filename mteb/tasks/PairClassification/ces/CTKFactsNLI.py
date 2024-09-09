from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CTKFactsNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CTKFactsNLI",
        dataset={
            "path": "ctu-aic/ctkfacts_nli",
            "revision": "387ae4582c8054cb52ef57ef0941f19bd8012abf",
            "trust_remote_code": True,
        },
        description="Czech Natural Language Inference dataset of around 3K evidence-claim pairs labelled with SUPPORTS, REFUTES or NOT ENOUGH INFO veracity labels. Extracted from a round of fact-checking experiments.",
        reference="https://arxiv.org/abs/2201.11115",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["ces-Latn"],
        main_score="max_ap",
        date=("2020-09-01", "2021-08-31"),  # academic year 2020/2021
        domains=["News", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{ullrich2023csfever,
        title={CsFEVER and CTKFacts: acquiring Czech data for fact verification},
        author={Ullrich, Herbert and Drchal, Jan and R{\`y}par, Martin and Vincourov{\'a}, Hana and Moravec, V{\'a}clav},
        journal={Language Resources and Evaluation},
        volume={57},
        number={4},
        pages={1571--1605},
        year={2023},
        publisher={Springer}
        }""",  # after removing label 1=NOT ENOUGH INFO
        descriptive_stats={
            "n_samples": {
                "test": 375,
                "validation": 305,
            },
            "avg_character_length": {"test": 225.62, "validation": 219.32},
        },
    )

    def dataset_transform(self):
        _dataset = {}
        self.dataset.pop("train")
        # keep labels 0=REFUTES and 2=SUPPORTS, and map them as 0 and 1 for binary classification
        hf_dataset = self.dataset.filter(lambda x: x["label"] in [0, 2])
        hf_dataset = hf_dataset.map(
            lambda example: {"label": 1 if example["label"] == 2 else 0}
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": hf_dataset[split]["evidence"],
                    "sentence2": hf_dataset[split]["claim"],
                    "labels": hf_dataset[split]["label"],
                }
            ]
        self.dataset = _dataset
