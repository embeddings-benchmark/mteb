from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedicalQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedicalQARetrieval",
        description="The dataset consists 2048 medical question and answer pairs.",
        reference="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4",
        dataset={
            "path": "mteb/medical_qa",
            "revision": "ae763399273d8b20506b80cf6f6f9a31a6a2b238",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2019-12-31"),  # best guess,
        domains=["Medical", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@ARTICLE{BenAbacha-BMC-2019,
            author    = {Asma, Ben Abacha and Dina, Demner{-}Fushman},
            title     = {A Question-Entailment Approach to Question Answering},
            journal = {{BMC} Bioinform.},
            volume    = {20},
            number    = {1},
            pages     = {511:1--511:23},
            year      = {2019},
            url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4}
            } """,
    )
