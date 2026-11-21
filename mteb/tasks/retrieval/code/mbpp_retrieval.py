from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MBPPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MBPPRetrieval",
        description="A code retrieval task based on 378 Python programming problems from MBPP (Mostly Basic Python Programming). Each query is a natural language description of a programming task (e.g., 'Write a function to find the shared elements from the given two lists'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper syntax and logic.",
        reference="https://huggingface.co/datasets/embedding-benchmark/MBPP",
        dataset={
            "path": "mteb/MBPPRetrieval",
            "revision": "2c7c0ff055f2e546880d8dab7d689413348683b6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{austin2021program,
  author = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal = {arXiv preprint arXiv:2108.07732},
  title = {Program Synthesis with Large Language Models},
  year = {2021},
}
""",
    )
