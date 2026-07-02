from datasets import DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakRTE(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakRTE",
        dataset={
            "path": "slovak-nlp/sklep",
            "revision": "10549a8c63542e6a0db4fcf5fcdc29b3e1b8c4e9",
            "name": "rte",
        },
        description="Slovak Recognizing Textual Entailment dataset. Professional translation and human verification of English RTE datasets (combining multiple sources) for Slovak language. The task is binary classification (entailment vs. not entailment).",
        reference="https://aclanthology.org/2025.findings-acl.1371",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="max_ap",
        date=("2025-01-01", "2025-04-30"),
        domains=["News", "Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{suppa-etal-2025-sklep,
  address = {Vienna, Austria},
  author = {Suppa, Marek and Ridzik, Andrej and Hl{\'a}dek, Daniel and Jav{\r{u}}rek, Tom{\'a}{\v{s}} and Ondrejov{\'a}, Vikt{\'o}ria and S{\'a}sikov{\'a}, Krist{\'i}na and Tamajka, Martin and Simko, Marian},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-256-5},
  month = jul,
  pages = {26716--26743},
  publisher = {Association for Computational Linguistics},
  title = {sk{LEP}: A {S}lovak General Language Understanding Benchmark},
  url = {https://aclanthology.org/2025.findings-acl.1371/},
  year = {2025},
}
""",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
    )

    def dataset_transform(self, **kwargs):
        _dataset = {}

        for split in self.dataset:
            hf_dataset = self.dataset[split]

            entailment_idx = hf_dataset.features["label"].names.index("entailment")
            labels = [
                1 if label == entailment_idx else 0 for label in hf_dataset["label"]
            ]

            _dataset[split] = [
                {
                    "sentence1": hf_dataset["text1"],
                    "sentence2": hf_dataset["text2"],
                    "labels": labels,
                }
            ]

        self.dataset = DatasetDict(_dataset)
