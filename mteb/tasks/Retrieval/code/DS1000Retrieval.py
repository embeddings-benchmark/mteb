from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DS1000Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DS1000Retrieval",
        description="A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.",
        reference="https://huggingface.co/datasets/embedding-benchmark/DS1000",
        dataset={
            "path": "embedding-benchmark/DS1000",
            "revision": "25cd4dc8172e799235d83c66439b6b7b8e6583ec",
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
