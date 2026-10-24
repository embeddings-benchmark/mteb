from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DS1000Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DS1000Retrieval",
        description="A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.",
        reference="https://huggingface.co/datasets/embedding-benchmark/DS1000",
        dataset={
            "path": "mteb/DS1000Retrieval",
            "revision": "26779eaf6daa2b7984cb33ec450817fd188aab30",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lai2022ds,
  author = {Lai, Yuhang and Li, Chengxi and Wang, Yiming and Zhang, Tianyi and Zhong, Ruiqi and Zettlemoyer, Luke and Yih, Wen-tau and Fried, Daniel and Wang, Sida and Yu, Tao},
  journal = {arXiv preprint arXiv:2211.11501},
  title = {DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  year = {2022},
}
""",
    )
