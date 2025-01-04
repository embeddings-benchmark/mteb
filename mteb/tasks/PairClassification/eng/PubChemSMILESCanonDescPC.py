from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PubChemSMILESCanonDescPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PubChemSMILESCanonDescPC",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/PubChemSMILESCanonDescPC",
            "revision": "6236cc0c3003bea6034d00f96d2202f7c05629c6"
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Chemistry"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
        @article{kasmaee2024chemteb,
        title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
        author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
        journal={arXiv preprint arXiv:2412.00532},
        year={2024}
        }
        """,
    )

    def load_data(self):
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
                    "sentence1": hf_dataset["description"],
                    "sentence2": hf_dataset["canonical_smiles"],
                    "labels": hf_dataset["labels"]
                }
            ]
        self.dataset = _dataset
