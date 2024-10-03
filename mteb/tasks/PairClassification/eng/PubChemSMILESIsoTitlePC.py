from __future__ import annotations


from typing import Any

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


_DATASET_COLUMN_MAP = {
    "sentence1": "title",
    "sentence2": "isomeric_smiles",
    "labels": "labels",
}


class PubChemSMILESIsoTitlePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PubChemSMILESIsoTitlePC",
        description="""TBW""",
        reference="https://pubchem.ncbi.nlm.nih.gov/",
        dataset={
            "path": "BASF-We-Create-Chemistry/PubChemSMILESIsoTitlePC",
            "revision": "1b0d57516ec7c168b8da44f80148b5418ba394b3"
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_f1",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation="created",
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": {"train": 43052},
            "avg_character_length": {"train": 34}
        }
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        _dataset = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            trust_remote_code=True,
        )

        self.dataset = _dataset
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=self.metadata_dict["eval_splits"], label="labels"
        )

        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset[_DATASET_COLUMN_MAP["sentence1"]],
                    "sentence2": hf_dataset[_DATASET_COLUMN_MAP["sentence2"]],
                    "labels": hf_dataset[_DATASET_COLUMN_MAP["labels"]],
                }
            ]
        self.dataset = _dataset
