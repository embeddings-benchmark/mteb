from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_DATASET_COLUMN_MAP = [
    {
        "name": "citation_prediction_classification",
        "sent1": "citation",
        "sent2": "text",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "consumer_contracts_qa",
        "sent1": "question",
        "sent2": "contract",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "contract_qa",
        "sent1": "question",
        "sent2": "text",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "hearsay",
        "sent1": "text",
        "sent2": "slice",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "privacy_policy_entailment",
        "sent1": "text",
        "sent2": "description",
        "labels": "answer",
        "mapping": {"correct": 1, "incorrect": 0},
    },
    {
        "name": "privacy_policy_qa",
        "sent1": "text",
        "sent2": "question",
        "labels": "answer",
        "mapping": {"relevant": 1, "irrelevant": 0},
    },
]


class LegalBenchPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LegalBenchPC",
        description="""This LegalBench pair classification task is a combination of the following datasets:

        - Citation Prediction Classification: Given a legal statement and a case citation, determine if the citation is supportive of the legal statement.
        - Consumer Contracts QA: The task consists of 400 yes/no questions relating to consumer contracts (specifically, online terms of service) and is relevant to the legal skill of contract interpretation.
        - Contract QA: Answer yes/no questions about whether contractual clauses discuss particular issues like confidentiality requirements, BIPA consent, PII data breaches, breach of contract etc.
        - Hearsay: Classify if a particular piece of evidence qualifies as hearsay. Each sample in the dataset describes (1) an issue being litigated or an assertion a party wishes to prove, and (2) a piece of evidence a party wishes to introduce. The goal is to determine if—as it relates to the issue—the evidence would be considered hearsay under the definition provided above.
        - Privacy Policy Entailment: Given a privacy policy clause and a description of the clause, determine if the description is correct. This is a binary classification task in which the LLM is provided with a clause from a privacy policy, and a description of that clause (e.g., “The policy describes collection of the user’s HTTP cookies, flash cookies, pixel tags, or similar identifiers by a party to the contract.”).
        - Privacy Policy QA: Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question. This is a binary classification task in which the LLM is provided with a question (e.g., “do you publish my data”) and a clause from a privacy policy. The LLM must determine if the clause contains an answer to the question, and classify the question-clause pair.
        """,
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
            "trust_remote_code": True,
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }
        @article{kolt2022predicting,
            title={Predicting consumer contracts},
            author={Kolt, Noam},
            journal={Berkeley Tech. LJ},
            volume={37},
            pages={71},
            year={2022},
            publisher={HeinOnline}
        }
        @article{zimmeck2019maps,
            title={Maps: Scaling privacy compliance analysis to a million apps},
            author={Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
            journal={Proc. Priv. Enhancing Tech.},
            volume={2019},
            pages={66},
            year={2019}
        }
        @article{ravichander2019question,
            title={Question answering for privacy policies: Combining computational and legal perspectives},
            author={Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
            journal={arXiv preprint arXiv:1911.00841},
            year={2019}
        }
        """,
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
                    dataset_col_map["sent1"]: "sentence1",
                    dataset_col_map["sent2"]: "sentence2",
                    dataset_col_map["labels"]: "labels",
                }
            )
            _dataset = _dataset.select_columns(["labels", "sentence1", "sentence2"])
            mapping = dataset_col_map["mapping"]
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
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label="labels"
        )

        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["sentence1"],
                    "sentence2": hf_dataset["sentence2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
