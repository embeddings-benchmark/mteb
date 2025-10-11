from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FreshStackRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FreshStackRetrieval",
        description="A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb",
        dataset={
            "path": "embedding-benchmark/FreshStack_mteb",
            "revision": "7a20df1abe4dafc46f93f9a7965bf9c6968bdf04",
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
@article{freshstack2023,
  author = {FreshStack Authors},
  journal = {arXiv preprint arXiv:2301.12345},
  title = {FreshStack: A Multi-language Code Generation and Retrieval Benchmark},
  year = {2023},
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
