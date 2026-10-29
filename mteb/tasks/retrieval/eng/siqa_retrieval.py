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
@inproceedings{sap2019socialiqa,
  address = {Hong Kong, China},
  author = {Sap, Maarten and Rashkin, Hannah and Chen, Derek and Le Bras, Ronan and Choi, Yejin},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing ({EMNLP}-{IJCNLP})},
  doi = {10.18653/v1/D19-1454},
  editor = {Inui, Kentaro and Jiang, Jing and Ng, Vincent and Wan, Xiaojun},
  month = {nov},
  pages = {4463--4473},
  publisher = {Association for Computational Linguistics},
  title = {Social {IQ}a: Commonsense Reasoning about Social Interactions},
  url = {https://aclanthology.org/D19-1454/},
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
