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
@inproceedings{jimenez2024swebenchlanguagemodelsresolve,
  archiveprefix = {arXiv},
  author = {Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2310.06770},
  title = {{SWE}-bench: Can Language Models Resolve Real-world Github Issues?},
  url = {https://openreview.net/forum?id=VTF8yNQM66},
  year = {2024},
}
""",
    )
