from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEbenchMultilingualRR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEbenchMultilingualRR",
        description="Multilingual Software Issue Localization.",
        reference="https://www.swebench.com/multilingual.html",
        dataset={
            "path": "mteb/SWEbenchMultilingualRR",
            "revision": "2129d0fd2fe31e95b4eecbeca873b623d7f58c4d",
        },
        type="Reranking",
        category="t2t",
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
@inproceedings{yang2025swesmith,
  archiveprefix = {arXiv},
  author = {John Yang and Kilian Lieret and Carlos E Jimenez and Alexander Wettig and Kabir Khandpur and Yanzhe Zhang and Binyuan Hui and Ofir Press and Ludwig Schmidt and Diyi Yang},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  eprint = {2504.21798},
  title = {{SWE}-smith: Scaling Data for Software Engineering Agents},
  url = {https://openreview.net/forum?id=63iVrXc8cC},
  year = {2026},
}
""",
    )
