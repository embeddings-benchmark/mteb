from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from .BIRCOBase import BIRCOBase

class BIRCOClinicalTrialReranking(BIRCOBase):
    metadata = TaskMetadata(
        name="BIRCOClinicalTrialReranking",
        description=(
            "BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives.\n\n"
            "Clinical‑Trial: In this task, a query is a patient case report, and each candidate passage is a clinical trial description. "
            "Relevance is rated on a scale of 0, 1, or 2; scores ≥1 indicate a positive match."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        dataset={
            "path": "bpHigh/BIRCO_Clinical_Trial",
            "revision": "YOUR_REVISION_PLACEHOLDER"  # Placeholder: update with actual revision.
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        _needs_graded_evaluation=True,
        main_score="ndcg_at_10",
        date=("2024-04-03", "2024-04-03"),  # Placeholder: update dates if needed.
        form=["written"],
        domains=["Medical"],
        task_subtypes=["Clinical Trial Matching"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{birco2024clinicaltrial,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
            author={Wang, Xiaoyue and others},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR}
        }""",
        n_samples={"test": 100},
        avg_character_length={"test": 900}
    )

    def get_query(self, sample):
        instruction = (
            "Instruction: In this task, a query is a patient case report and the candidate passages are clinical trial descriptions. "
            "The objective is to match eligible patients (query) to appropriate clinical trials (passages)."
        )
        return instruction + "\n" + sample["query"]

    def get_positive_docs(self, sample):
        return sample["positive"]

    def get_negative_docs(self, sample):
        return sample["negative"]
