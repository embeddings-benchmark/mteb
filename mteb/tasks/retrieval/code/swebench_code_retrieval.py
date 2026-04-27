from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEbenchCodeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEbenchCodeRetrieval",
        description="A code retrieval task based on SWE-bench Verified, a curated set of 500 real GitHub issues from 12 popular Python repositories. Each query is a GitHub issue description (bug report or feature request), and the corpus contains Python source files from the associated repositories. The task is to retrieve the source files that need to be modified to resolve each issue. This represents a realistic software engineering retrieval scenario where developers search codebases to locate relevant files for bug fixes.",
        reference="https://www.swebench.com/",
        dataset={
            "path": "mteb/SWEbenchCodeRetrieval",
            "revision": "6b453d3a65280200d1931c7edd3e64be230cd69b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2023-10-10", "2024-12-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a GitHub issue, retrieve the source code files that need to be modified to resolve the issue."
        },
        bibtex_citation=r"""
@misc{jimenez2024swebenchlanguagemodelsresolve,
  archiveprefix = {arXiv},
  author = {Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
  eprint = {2310.06770},
  primaryclass = {cs.CL},
  title = {SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  url = {https://arxiv.org/abs/2310.06770},
  year = {2024},
}
""",
    )
