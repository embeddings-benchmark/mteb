from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MultiSWEbenchReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MultiSWEbenchRR",
        description="Multilingual Software Issue Localization.",
        reference="https://multi-swe-bench.github.io/#/",
        dataset={
            "path": "tarsur909/mteb-swe-bench-multi-reranking",
            "revision": "38e01d5155c9f885942ea6d69e6722e74b9d4013",
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
@misc{zan2025multiswebench,
  archiveprefix = {arXiv},
  author = {Daoguang Zan and Zhirong Huang and Wei Liu and Hanwu Chen and Linhao Zhang and Shulin Xin and Lu Chen and Qi Liu and Xiaojian Zhong and Aoyan Li and Siyao Liu and Yongsheng Xiao and Liangqiang Chen and Yuyu Zhang and Jing Su and Tianyu Liu and Rui Long and Kai Shen and Liang Xiang},
  eprint = {2504.02605},
  primaryclass = {cs.SE},
  title = {Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving},
  url = {https://arxiv.org/abs/2504.02605},
  year = {2025},
}
""",
    )
