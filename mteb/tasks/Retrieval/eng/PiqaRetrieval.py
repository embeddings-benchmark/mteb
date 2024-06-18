from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PIQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.",
        reference="https://arxiv.org/abs/1911.11641",
        dataset={
            "path": "RAR-b/piqa",
            "revision": "bb30be7e9184e6b6b1d99bbfe1bb90a3a81842e6",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval"],
        license="AFL-3.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@inproceedings{bisk2020piqa,
  title={Piqa: Reasoning about physical commonsense in natural language},
  author={Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={05},
  pages={7432--7439},
  year={2020}
}
""",
        n_samples={"test": 1838},
        avg_character_length={"test": 134.3},
    )
