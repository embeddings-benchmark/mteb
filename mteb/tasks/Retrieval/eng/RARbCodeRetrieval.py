from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RARbCode(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbCode",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "RAR-b/humanevalpack-mbpp-pooled",
            "revision": "25f7d11a7ac12dcbb8d3836eb2de682b98c825e4",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{husain2019codesearchnet,
  author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal = {arXiv preprint arXiv:1909.09436},
  title = {Codesearchnet challenge: Evaluating the state of semantic code search},
  year = {2019},
}

@article{muennighoff2023octopack,
  author = {Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and Von Werra, Leandro and Longpre, Shayne},
  journal = {arXiv preprint arXiv:2308.07124},
  title = {Octopack: Instruction tuning code large language models},
  year = {2023},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={"query": "Retrieve the answer for the following coding problem."},
    )
