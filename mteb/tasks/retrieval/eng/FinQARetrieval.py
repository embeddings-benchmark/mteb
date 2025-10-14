from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FinQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinQARetrieval",
        description="A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FinQA",
        dataset={
            "path": "embedding-benchmark/FinQA",
            "revision": "bdd1903ce03153129480bfc14b710e3d612c1efd",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chen2021finqa,
  author = {Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  journal = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  title = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
  year = {2021},
}
""",
        prompt={
            "query": "Given a financial numerical reasoning question, retrieve relevant financial data that helps answer the question"
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
