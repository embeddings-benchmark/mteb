from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FinanceBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBenchRetrieval",
        description="A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FinanceBench",
        dataset={
            "path": "mteb/FinanceBenchRetrieval",
            "revision": "c5215673492044833ff5e0dfd9b5fa848d5df0f2",
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
