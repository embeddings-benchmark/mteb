from datasets import DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakNLI",
        description="Slovak Handwritten Annotated NLI dataset",
        reference="https://huggingface.co/datasets/natalia-nk/NLI-SK-annotated",
        dataset={
            "path": "natalia-nk/NLI-SK-annotated",
            "revision": "79914f425b59d8b9fabb6d38c37f6d81f9723f46",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        date=("2024-10-01", "2025-07-31"),
        domains=["News", "Web", "Written"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="max_ap",
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
        bibtex_citation="",
    )

    def dataset_transform(self):
        _dataset = {}

        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split].filter(
                lambda x: x["Label"] in ["Entailment", "Contradiction"]
            )
            hf_dataset = hf_dataset.map(
                lambda example: {"Label": 1 if example["Label"] == "Entailment" else 0}
            )

            _dataset[split] = [
                {
                    "sentence1": hf_dataset["Premise"],
                    "sentence2": hf_dataset["Hypothesis"],
                    "labels": hf_dataset["Label"],
                }
            ]

        self.dataset = DatasetDict(_dataset)
