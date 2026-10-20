from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RARbCode(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbCode",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "mteb/RARbCode",
            "revision": "ed0a01c3e4d39ec17badb87405e595322f8f2b86",
        },
        type="Retrieval",
        category="t2t",
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
  title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
  year = {2019},
}

@inproceedings{muennighoff2023octopack,
  archiveprefix = {arXiv},
  author = {Niklas Muennighoff and Qian Liu and Armel Randy Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro Von Werra and Shayne Longpre},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2308.07124},
  title = {OctoPack: Instruction Tuning Code Large Language Models},
  url = {https://openreview.net/forum?id=mw1PWNSWZP},
  year = {2024},
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
