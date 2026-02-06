from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LawIRKo(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LawIRKo",
        description="""QA evaluation dataset based on Korean legal domain
        Task: The primary objective of this dataset is to evaluate a model's retrieval performance in identifying the relevant legal text when provided with a query containing specific information about a law and its articles.
        Documents: The document corpus consists of official Korean legal and statutory texts, including statutes, acts, and regulations. Each document represents an individual legal article.
        Queries: The queries are reconstructed based on specific law titles and their corresponding article names (e.g., 2: Responsibilities of Personal Information Controllers)
        """,
        reference="https://huggingface.co/datasets/on-and-on/lawgov_ir-ko",
        dataset={
            "path": "on-and-on/lawgov_ir-ko",
            "revision": "bd5361e486ef4be7052c506adfdf0610d04abbfe",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-01", "2026-02-01"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{law_ko_ir_khee,
  author = {kang-hyeun Lee},
  howpublished = {\url{https://huggingface.co/datasets/on-and-on/lawgov_ir-ko}},
  note = {A Benchmark Dataset for Korean Legal Information Retrieval and QA},
  year = {2026},
}
""",
    )
