from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class PIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PIQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.",
        reference="https://arxiv.org/abs/1911.11641",
        dataset={
            "path": "mteb/PIQA",
            "revision": "aee23bb32dafbaddddf16cd5033d912693984d9a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="afl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{bisk2020piqa,
  author = {Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
  booktitle = {Proceedings of the AAAI conference on artificial intelligence},
  number = {05},
  pages = {7432--7439},
  title = {Piqa: Reasoning about physical commonsense in natural language},
  volume = {34},
  year = {2020},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={"query": "Given the following goal, retrieve a possible solution."},
    )
