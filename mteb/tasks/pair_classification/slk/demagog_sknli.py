from datasets import DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DemagogSKNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="DemagogSKNLI",
        description=(
            "Slovak Natural Language Inference dataset created from Demagog.sk fact-checking data. "
            "The dataset consists of evidence-claim pairs where professional fact-checkers' analysis "
            "(evidence) is paired with political statements (claims). Labels indicate whether the "
            "evidence supports or refutes the claim. Only clear verdicts (Pravda/Nepravda) are included, "
            "filtering out ambiguous cases (Zavádzajúce/Neoveriteľné)."
        ),
        reference="https://huggingface.co/datasets/NaiveNeuron/DemagogSK",
        dataset={
            "path": "NaiveNeuron/DemagogSK",
            "revision": "0f9e0f902da3aca1fd6c940cd852e95c9ef5327f",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        date=("2010-03-14", "2025-10-16"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="max_ap",
        domains=["Government", "News", "Written"],
        task_subtypes=["Claim verification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Given a fact-checker's analysis (evidence), determine if it supports or refutes the political claim",
    )

    def dataset_transform(self):
        """Transform DemagogSK into evidence-claim NLI pairs."""
        _dataset = {}

        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]

            # Filter to only include clear verdicts (Pravda/Nepravda)
            # and ensure both evidence and claim are present
            hf_dataset = hf_dataset.filter(
                lambda x: (
                    x.get("verdict") in ["Pravda", "Nepravda"]
                    and x.get("analysis_text")
                    and x.get("statement")
                    and len(str(x.get("analysis_text", "")).strip()) > 0
                    and len(str(x.get("statement", "")).strip()) > 0
                )
            )

            # Map verdicts: Pravda (true) -> 1 (SUPPORTS), Nepravda (false) -> 0 (REFUTES)
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 1 if example["verdict"] == "Pravda" else 0}
            )

            _dataset[split] = [
                {
                    "sentence1": hf_dataset["analysis_text"],  # Evidence/reasoning
                    "sentence2": hf_dataset["statement"],  # Claim to verify
                    "labels": hf_dataset["label"],
                }
            ]

        self.dataset = DatasetDict(_dataset)
