from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CTKFactsNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CTKFactsNLI",
        dataset={
            "path": "ctu-aic/ctkfacts_nli",
            "revision": "387ae4582c8054cb52ef57ef0941f19bd8012abf",
        },
        description="Czech Natural Language Inference dataset of around 3K evidence-claim pairs labelled with SUPPORTS, REFUTES or NOT ENOUGH INFO veracity labels. Extracted from a round of fact-checking experiments.",
        reference="https://arxiv.org/abs/2201.11115",
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ces-Latn"],
        main_score="ap",
        date=("2020-09-01", "2021-08-31"),  # academic year 2020/2021
        form=["written"],
        domains=["News"],
        task_subtypes=["Claim verification"],
        license="CC-BY-SA-3.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{ullrich2023csfever,
        title={CsFEVER and CTKFacts: acquiring Czech data for fact verification},
        author={Ullrich, Herbert and Drchal, Jan and R{\`y}par, Martin and Vincourov{\'a}, Hana and Moravec, V{\'a}clav},
        journal={Language Resources and Evaluation},
        volume={57},
        number={4},
        pages={1571--1605},
        year={2023},
        publisher={Springer}
        }""",
        n_samples={"validation": 375},  # after removing NOT ENOUGH INFO
        avg_character_length={"validation": 225.62},
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            # keep labels 0=REFUTES and 2=SUPPORTS, and map them as 0 and 1 for binary classification
            hf_dataset = self.dataset[split].filter(lambda x: x["label"] in [0, 2])
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 1 if example["label"] == 2 else 0}
            )
            _dataset[split] = [
                {
                    "sent1": hf_dataset["evidence"],
                    "sent2": hf_dataset["claim"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
