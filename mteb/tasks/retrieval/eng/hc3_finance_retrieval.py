from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HC3FinanceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3FinanceRetrieval",
        description="A financial retrieval task based on HC3 Finance dataset containing human vs AI-generated financial text detection. Each query is a financial question or prompt (e.g., 'Explain the impact of interest rate changes on bond prices'), and the corpus contains both human-written and AI-generated financial responses. The task is to retrieve the most relevant and accurate financial content that addresses the query. Queries are financial questions while the corpus contains detailed financial explanations, analysis, and educational content covering various financial concepts and market dynamics.",
        reference="https://huggingface.co/datasets/embedding-benchmark/HC3Finance",
        dataset={
            "path": "mteb/HC3FinanceRetrieval",
            "revision": "f7674417d3552fd20e071762bff049137f17b075",
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
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{guo2023hc3,
  author = {Guo, Biyang and Zhang, Xin and Wang, Zhiyuan and Jiang, Mingyuan and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
  journal = {arXiv preprint arXiv:2301.07597},
  title = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  year = {2023},
}
""",
        prompt={
            "query": "Given a financial question or prompt, retrieve relevant financial content that best addresses the query"
        },
    )
