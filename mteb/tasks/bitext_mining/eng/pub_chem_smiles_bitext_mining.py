from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

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


class PubChemSMILESBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="PubChemSMILESBitextMining",
        dataset={
            "path": "BASF-AI/PubChemSMILESBitextMining",
            "revision": "36700ea628118312ebf2f90ad2353a9a8f188dc9",
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html",
        type="BitextMining",
        category="t2t",
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
@inproceedings{kasmaee2024chemteb,
  archiveprefix = {arXiv},
  author = {Ali Shiraee Kasmaee and Mohammad Khodadad and Mohammad Arshi Saloot and Nick Sherck and Stephen Dokas and Hamidreza Mahyar and Soheila Samiee},
  booktitle = {ENLSP},
  eprint = {2412.00532},
  pages = {512-531},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance & Efficiency on a Specific Domain},
  url = {https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html},
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

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        for subset in self.hf_subsets:
            self.dataset[subset] = self.dataset[subset].rename_columns(
                COL_MAPPING[subset]
            )
