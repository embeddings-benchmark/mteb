from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WikiSQLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikiSQLRetrieval",
        description="A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.",
        reference="https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb",
        dataset={
            "path": "mteb/WikiSQLRetrieval",
            "revision": "add6c3faf0d972ebb139b30fa6c4f955d99635c5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "sql-Code"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="bsd-3-clause",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{zhong2017seq2sql,
  archiveprefix = {arXiv},
  author = {Zhong, Victor and Xiong, Caiming and Socher, Richard},
  eprint = {1709.00103},
  primaryclass = {cs.CL},
  title = {Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning},
  year = {2017},
}
""",
    )
