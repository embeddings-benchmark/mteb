from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SWEbenchVerifiedReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SWEbenchVerifiedRR",
        description="Software Issue Localization for SWE-bench Verified",
        reference="https://openai.com/index/introducing-swe-bench-verified/",
        dataset={
            "path": "tarsur909/mteb-swe-bench-verified-reranking",
            "revision": "796ae0b4b187e5c0533a12411dee0d8e34eaf0b5",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2024-08-13", "2024-08-13"),  # arxiv v1 submission date
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a github issue, identify the code that needs to be changed to fix the issue."
        },
        bibtex_citation=r"""
@misc{openai2024swebenchverified,
  author = {OpenAI},
  title = {Introducing swe-bench verified},
  url = {https://openai.com/index/introducing-swe-bench-verified/},
  year = {2024},
}
""",
    )
