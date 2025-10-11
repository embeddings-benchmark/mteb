from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WikiSQLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikiSQLRetrieval",
        description="A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.",
        reference="https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb",
        dataset={
            "path": "embedding-benchmark/WikiSQL_mteb",
            "revision": "4e099ab42dffd49d72c1472f451371e53343e3d7",
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

    def load_data(self) -> None:
        if self.data_loaded:
            return

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(
            self.metadata.dataset["path"],
            "corpus",
            revision=self.metadata.dataset["revision"],
        )["corpus"]
        queries_ds = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            revision=self.metadata.dataset["revision"],
        )["queries"]
        qrels_ds = load_dataset(
            self.metadata.dataset["path"],
            "default",
            revision=self.metadata.dataset["revision"],
        )["test"]

        # Initialize data structures with 'test' split
        corpus = {}
        queries = {}
        relevant_docs = {}

        # Process corpus
        for item in corpus_ds:
            corpus[item["id"]] = {"title": "", "text": item["text"]}

        # Process queries
        for item in queries_ds:
            queries[item["id"]] = item["text"]

        # Process qrels (relevant documents)
        for item in qrels_ds:
            query_id = item["query-id"]
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][item["corpus-id"]] = int(item["score"])

        # Organize data by splits as expected by MTEB
        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}

        self.data_loaded = True
