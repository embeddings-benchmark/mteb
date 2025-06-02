from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_DATASET_COLUMN_MAP = [
    {
        "name": "iso-desc",
        "sent1": "description",
        "sent2": "isomeric_smiles",
        "labels": "labels",
    },
    {
        "name": "iso-title",
        "sent1": "title",
        "sent2": "isomeric_smiles",
        "labels": "labels",
    },
    {
        "name": "canon-desc",
        "sent1": "description",
        "sent2": "canonical_smiles",
        "labels": "labels",
    },
    {
        "name": "canon-title",
        "sent1": "title",
        "sent2": "canonical_smiles",
        "labels": "labels",
    },
]


class PubChemSMILESPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PubChemSMILESPC",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/PubChemSMILESPairClassification",
            "revision": "7ba40b69f5fe6ffe4cc189aac9e1710913c73c8a",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}

@article{kim2023pubchem,
  author = {Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and others},
  journal = {Nucleic acids research},
  number = {D1},
  pages = {D1373--D1380},
  publisher = {Oxford University Press},
  title = {PubChem 2023 update},
  volume = {51},
  year = {2023},
}
""",
    )

    def load_data(self):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        _hf_dataset = None
        for dataset_col_map in _DATASET_COLUMN_MAP:
            _dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                dataset_col_map["name"],
                revision=self.metadata.dataset["revision"],
            )

            _dataset = _dataset.rename_columns(
                {
                    dataset_col_map["sent1"]: "sentence1",
                    dataset_col_map["sent2"]: "sentence2",
                    dataset_col_map["labels"]: "labels",
                }
            )

            if _hf_dataset is None:
                _hf_dataset = _dataset
            else:
                _hf_dataset["test"] = datasets.concatenate_datasets(
                    [_hf_dataset["test"], _dataset["test"]]
                )

        self.dataset = _hf_dataset
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=self.metadata["eval_splits"],
            label="labels",
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
