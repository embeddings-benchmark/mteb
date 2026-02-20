from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HellaSwag(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HellaSwag",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.",
        reference="https://rowanzellers.com/hellaswag/",
        dataset={
            "path": "mteb/HellaSwag",
            "revision": "944be88c186b87495f4b1f5e4a416a3a7ea5d680",
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
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}

@article{zellers2019hellaswag,
  author = {Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
  journal = {arXiv preprint arXiv:1905.07830},
  title = {Hellaswag: Can a machine really finish your sentence?},
  year = {2019},
}
""",
        prompt={
            "query": "Given the following unfinished context, retrieve the most plausible ending to finish it."
        },
    )
