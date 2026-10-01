from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FinQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinQARetrieval",
        description="A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.",
        reference="https://huggingface.co/datasets/embedding-benchmark/FinQA",
        dataset={
            "path": "mteb/FinQARetrieval",
            "revision": "18dbd07c4e9dc8328d10a53b98f3caad6b7e07d9",
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
