from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LawIRKo(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="law_ir-ko",
        description="law_ir-ko",
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
        date="2026-02-01",
        domains=["legal", "korean_law"],
        task_subtypes=["legal_qa"],
        license="mit",
        annotations_creators="human-verified",
        dialect=[],
        sample_creation="curated from National Law Information Center",
        bibtex_citation=r"""
@misc{law_ko_ir_on,
  author = {on-and-on},
  note = {A Benchmark Dataset for Korean Legal Information Retrieval and QA},
  howpublished = {\url{https://www.law.go.kr/LSW/main.html}},
  year = {2026},
}
""",
    )
