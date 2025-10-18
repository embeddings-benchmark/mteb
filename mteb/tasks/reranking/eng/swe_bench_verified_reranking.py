from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEbenchVerifiedReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEbenchVerifiedRR",
        description="Software Issue Localization for SWE-bench Verified",
        reference="https://openai.com/index/introducing-swe-bench-verified/",
        dataset={
            "path": "mteb/SWEbenchVerifiedRR",
            "revision": "373818dbe743204d007c3c27ca091a3349331afa",
        },
        type="Reranking",
        category="t2t",
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
