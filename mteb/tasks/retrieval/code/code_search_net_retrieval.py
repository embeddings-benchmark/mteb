from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CodeSearchNetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeSearchNetRetrieval",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://huggingface.co/datasets/code_search_net/",
        dataset={
            "path": "mteb/CodeSearchNetRetrieval",
            "revision": "68e8f0731a656fa4bd5b7c81936d95ad48a39bfe",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            lang: [lang + "-Code"]
            for lang in ["python", "javascript", "go", "ruby", "java", "php"]
        },
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{husain2019codesearchnet,
  author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal = {arXiv preprint arXiv:1909.09436},
  title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
  year = {2019},
}
""",
    )
