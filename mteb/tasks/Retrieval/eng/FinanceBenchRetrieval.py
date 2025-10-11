from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FinanceBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBenchRetrieval",
        description="A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FinanceBench",
        dataset={
            "path": "embedding-benchmark/FinanceBench",
            "revision": "e68478442112cae36b70a216f52cc2777acf0a7e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{islam2023financebench,
  author = {Islam, Pranab and Kannappan, Anand and Kiela, Douwe and Fergus, Rob and Ott, Myle and Wang, Sam and Garimella, Aparna and Garcia, Nino},
  journal = {arXiv preprint arXiv:2311.11944},
  title = {FinanceBench: A New Benchmark for Financial Question Answering},
  year = {2023},
}
""",
        prompt={
            "query": "Given a financial question, retrieve relevant financial information that best answers the question"
        },
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
