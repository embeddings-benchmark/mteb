from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

COL_MAPPING = {
    "iso-title": {"title": "sentence1", "isomeric_smiles": "sentence2"},
    "iso-desc": {"description": "sentence1", "isomeric_smiles": "sentence2"},
    "canon-title": {"title": "sentence1", "canonical_smiles": "sentence2"},
    "canon-desc": {"description": "sentence1", "canonical_smiles": "sentence2"},
}

EVAL_LANGS = {
    "iso-title": ["en-Latn", "eng-Latn"],
    "iso-desc": ["en-Latn", "eng-Latn"],
    "canon-title": ["en-Latn", "eng-Latn"],
    "canon-desc": ["en-Latn", "eng-Latn"],
}


class PubChemSMILESBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="PubChemSMILESBitextMining",
        dataset={
            "path": "BASF-AI/PubChemSMILESBitextMining",
            "revision": "36700ea628118312ebf2f90ad2353a9a8f188dc9"
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=EVAL_LANGS,
        main_score="f1",
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
        @article{kim2023pubchem,
        title={PubChem 2023 update},
        author={Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and others},
        journal={Nucleic acids research},
        volume={51},
        number={D1},
        pages={D1373--D1380},
        year={2023},
        publisher={Oxford University Press}
        }
        """,
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return
        self.dataset = {}

        for subset in self.hf_subsets:
            self.dataset[subset] = datasets.load_dataset(
                **self.metadata_dict["dataset"], name=subset)

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        for subset in self.hf_subsets:
            self.dataset[subset] = self.dataset[subset].rename_columns(COL_MAPPING[subset])
