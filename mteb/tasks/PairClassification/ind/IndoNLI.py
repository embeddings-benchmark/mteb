from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class IndoNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="indonli",
        dataset={
            "path": "afaji/indonli",
            "revision": "3c976110fc13596004dc36279fc4c453ff2c18aa",
            "trust_remote_code": True,
        },
        description="IndoNLI is the first human-elicited Natural Language Inference (NLI) dataset for Indonesian. IndoNLI is annotated by both crowd workers and experts.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test_expert"],
        eval_langs=["ind-Latn"],
        main_score="max_ap",
        date=("2021-01-01", "2021-11-01"),  # best guess
        domains=["Encyclopaedic", "Web", "News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{mahendra-etal-2021-indonli,
  address = {Online and Punta Cana, Dominican Republic},
  author = {Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  month = nov,
  pages = {10511--10527},
  publisher = {Association for Computational Linguistics},
  title = {{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian},
  url = {https://aclanthology.org/2021.emnlp-main.821},
  year = {2021},
}
""",
        # after removing neutral
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            # keep labels 0=entailment and 2=contradiction, and map them as 1 and 0 for binary classification
            hf_dataset = self.dataset[split].filter(lambda x: x["label"] in [0, 2])
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 0 if example["label"] == 2 else 1}
            )
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["premise"],
                    "sentence2": hf_dataset["hypothesis"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
