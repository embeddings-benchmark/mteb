from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

class BarExamQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BarExamQA",
        description="The dataset includes questions from multistate bar exams and answers sourced from expert annotations.",
        reference="https://reglab.github.io/legal-rag-benchmarks/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "isaacus/mteb-barexam-qa",
            "revision": "4246981",
        },
        date=("2024-08-14", "2025-07-18"), 
        domains=["Legal", "Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{zheng2025,
  author = {Zheng, Lucia and Guha, Neel and Arifov, Javokhir and Zhang, Sarah and Skreta, Michal and Manning, Christopher D. and Henderson, Peter and Ho, Daniel E.},
  title = {A Reasoning-Focused Legal Retrieval Benchmark},
  year = {2025},
  series = {CSLAW '25 (forthcoming)}
""",
)
