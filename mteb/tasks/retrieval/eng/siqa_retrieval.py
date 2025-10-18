from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SIQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.",
        reference="https://leaderboard.allenai.org/socialiqa/submissions/get-started",
        dataset={
            "path": "mteb/SIQA",
            "revision": "59898732ccfb72afca329ce110d6799676ef0f84",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{sap2019socialiqa,
  author = {Sap, Maarten and Rashkin, Hannah and Chen, Derek and LeBras, Ronan and Choi, Yejin},
  journal = {arXiv preprint arXiv:1904.09728},
  title = {Socialiqa: Commonsense reasoning about social interactions},
  year = {2019},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following context and question, retrieve the correct answer."
        },
    )
