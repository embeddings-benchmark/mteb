from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LawIRKo(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LawIRKo",
        description="QA evaluation dataset based on Korean legal domain",
        reference=None,
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
        domains=["Legal"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{law_ko_ir_on,
  author = {on-and-on},
  note = {A Benchmark Dataset for Korean Legal Information Retrieval and QA},
  howpublished = {\url{https://www.law.go.kr/LSW/main.html}},
  year = {2026},
}
""",
    )
