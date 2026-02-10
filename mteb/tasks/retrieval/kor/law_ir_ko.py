from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LawIRKo(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LawIRKo",
        description="""This dataset assesses a model's ability to retrieve relevant legal articles from queries referencing specific Korean laws and provisions. The corpus comprises official legal texts including statutes, acts, and regulations, with each document representing a single article. Queries are derived from law titles paired and article identifiers. For instance the law title might be "건축법" (Building Act) and the article name "기술적 기준" (Technical Standards), which would become "건축법에 명시된 법률 중에 '기술적 기준'에 대해 설명하고 있는 세부 항목은 무엇입니까?" ("Which specific articles in the Building Act explain the 'technical standards'?").""",
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
