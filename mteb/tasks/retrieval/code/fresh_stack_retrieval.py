from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FreshStackRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FreshStackRetrieval",
        description="A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb",
        dataset={
            "path": "mteb/FreshStackRetrieval",
            "revision": "6faa3e4ff1c7d31824c32ee0a9dd580aba8bad11",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code", "javascript-Code", "go-Code"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{thakur2025freshstackbuildingrealisticbenchmarks,
  archiveprefix = {arXiv},
  author = {Nandan Thakur and Jimmy Lin and Sam Havens and Michael Carbin and Omar Khattab and Andrew Drozdov},
  eprint = {2504.13128},
  primaryclass = {cs.IR},
  title = {FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents},
  url = {https://arxiv.org/abs/2504.13128},
  year = {2025},
}
""",
    )
