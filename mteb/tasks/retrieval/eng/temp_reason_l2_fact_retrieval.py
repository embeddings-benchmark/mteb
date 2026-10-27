from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TempReasonL2Fact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TempReasonL2Fact",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-fact.",
        reference="https://github.com/DAMO-NLP-SG/TempReason",
        dataset={
            "path": "mteb/TempReasonL2Fact",
            "revision": "59b6048796e2a6ae40e843e9fa5675f047ae39b4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{tan2023towards,
  address = {Toronto, Canada},
  author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
  booktitle = {Proceedings of the 61st Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2023.acl-long.828},
  editor = {Rogers, Anna and Boyd-Graber, Jordan and Okazaki, Naoaki},
  month = {jul},
  pages = {14820--14835},
  publisher = {Association for Computational Linguistics},
  title = {Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models},
  url = {https://aclanthology.org/2023.acl-long.828/},
  year = {2023},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following question and facts, retrieve the correct answer."
        },
    )
