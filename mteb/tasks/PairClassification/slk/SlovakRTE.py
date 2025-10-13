from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakRTE(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakRTE",
        dataset={
            "path": "slovak-nlp/sklep",
            "revision": "2e428d17ae4178dae14f6643785647fc9f30edaa",
            "data_dir": "rte",
        },
        description="Slovak Recognizing Textual Entailment dataset. Professional translation and human verification of English RTE datasets (combining multiple sources) for Slovak language. The task is binary classification (entailment vs. not entailment).",
        reference="https://aclanthology.org/2025.findings-acl.1371",
        type="PairClassification",
        category="s2s",
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
    title = "sk{LEP}: A {S}lovak General Language Understanding Benchmark",
    author = "Suppa, Marek  and
      Ridzik, Andrej  and
      Hl{\'a}dek, Daniel  and
      Jav{\r{u}}rek, Tom{\'a}{\v{s}}  and
      Ondrejov{\'a}, Vikt{\'o}ria  and
      S{\'a}sikov{\'a}, Krist{\'i}na  and
      Tamajka, Martin  and
      Simko, Marian",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1371/",
    pages = "26716--26743",
    ISBN = "979-8-89176-256-5",
    abstract = "In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at \url{https://github.com/slovak-nlp/sklep} in the hopes of fostering reproducibility and drive future research in Slovak NLU."
}
""",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
    )

    def dataset_transform(self):
        # Rename columns from text1/text2 to sentence1/sentence2 for MTEB compatibility
        self.dataset = self.dataset.rename_column("text1", "sentence1")
        self.dataset = self.dataset.rename_column("text2", "sentence2")
        # Rename label to labels (plural) for MTEB compatibility
        self.dataset = self.dataset.rename_column("label", "labels")
