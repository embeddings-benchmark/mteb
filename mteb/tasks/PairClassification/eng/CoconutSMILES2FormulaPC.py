from __future__ import annotations


from typing import Any

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


_DATASET_COLUMN_MAP = {
    "sentence1": "name",
    "sentence2": "canonical_smiles",
    "labels": "label",
}


class CoconutSMILES2FormulaPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CoconutSMILES2FormulaPC",
        description="""TBW""",
        reference="https://coconut.naturalproducts.net/",
        dataset={
            "path": "BASF-We-Create-Chemistry/CoconutSMILES2FormulaPC",
            "revision": "76bb5eaa5cd86e795dfeb734692dca619d2aed74"
        },
        type="PairClassification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_f1",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators="derived",
        dialect=None,
        sample_creation="created",
        bibtex_citation=None,
        descriptive_stats={}
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            trust_remote_code=True,
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=_DATASET_COLUMN_MAP["labels"]
        )
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset[_DATASET_COLUMN_MAP["sentence1"]],
                    "sentence2": hf_dataset[_DATASET_COLUMN_MAP["sentence2"]],
                    "labels": hf_dataset[_DATASET_COLUMN_MAP["labels"]]
                }
            ]
        self.dataset = _dataset
