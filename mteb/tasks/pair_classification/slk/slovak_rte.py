from datasets import DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakRTE(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakRTE",
        dataset={
            "path": "slovak-nlp/sklep",
            "revision": "2e428d17ae4178dae14f6643785647fc9f30edaa",
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
  abstract = {In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at \url{https://github.com/slovak-nlp/sklep} in the hopes of fostering reproducibility and drive future research in Slovak NLU.},
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

    def dataset_transform(self):
        _dataset = {}

        for split in self.dataset:
            hf_dataset = self.dataset[split]

            _dataset[split] = [
                {
                    "sentence1": hf_dataset["text1"],
                    "sentence2": hf_dataset["text2"],
                    "labels": hf_dataset["label"],
                }
            ]

        self.dataset = DatasetDict(_dataset)
