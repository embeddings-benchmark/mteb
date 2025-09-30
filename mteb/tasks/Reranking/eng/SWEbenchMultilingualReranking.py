from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SWEPolyBenchReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SWEbenchMultilingualRR",
        description="Multilingual Software Issue Localization.",
        reference="https://www.swebench.com/multilingual.html",
        dataset={
            "path": "tarsur909/mteb-swe-bench-multilingual-reranking",
            "revision": "fb24db6aa765413d83044a830606f5ae5996ce7c",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2023-10-10", "2023-10-10"),
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
@misc{yang2025swesmith,
  archiveprefix = {arXiv},
  author = {John Yang and Kilian Lieret and Carlos E. Jimenez and Alexander Wettig and Kabir Khandpur and Yanzhe Zhang and Binyuan Hui and Ofir Press and Ludwig Schmidt and Diyi Yang},
  eprint = {2504.21798},
  primaryclass = {cs.SE},
  title = {SWE-smith: Scaling Data for Software Engineering Agents},
  url = {https://arxiv.org/abs/2504.21798},
  year = {2025},
}
""",
    )
