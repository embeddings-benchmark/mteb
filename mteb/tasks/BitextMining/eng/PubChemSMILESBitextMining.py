from __future__ import annotations

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
    "iso-title": ["eng-Latn", "eng-Latn"],
    "iso-desc": ["eng-Latn", "eng-Latn"],
    "canon-title": ["eng-Latn", "eng-Latn"],
    "canon-desc": ["eng-Latn", "eng-Latn"],
}


class PubChemSMILESBitextMining(MultilingualTask, AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="PubChemSMILESBitextMining",
        dataset={
            "path": "BASF-AI/PubChemSMILESBitextMining",
            "revision": "36700ea628118312ebf2f90ad2353a9a8f188dc9",
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=EVAL_LANGS,
        main_score="f1",
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

    def dataset_transform(self):
        for subset in self.hf_subsets:
            self.dataset[subset] = self.dataset[subset].rename_columns(
                COL_MAPPING[subset]
            )
