from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class PubChemSMILESISoTitleBM(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="PubChemSMILESISoTitleBM",
        dataset={
            "path": "BASF-AI/PubChemSMILESIsoTitleBM",
            "revision": "d60f975694c1841e60a39518b80e157c145f0be1"
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "en-en": ["en-Latn", "eng-Latn"]
        },
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
        """,
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}

        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                **self.metadata_dict["dataset"])

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].rename_columns({
            "title": "sentence1",
            "isomeric_smiles": "sentence2"
            })
