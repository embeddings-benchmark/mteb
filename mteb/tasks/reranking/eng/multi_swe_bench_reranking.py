from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MultiSWEbenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiSWEbenchRR",
        description="Multilingual Software Issue Localization.",
        reference="https://multi-swe-bench.github.io/#/",
        dataset={
            "path": "mteb/MultiSWEbenchRR",
            "revision": "f80517681e98077bb9b20470ab562bb6e94e7897",
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
@inproceedings{zan2025multiswebench,
  archiveprefix = {arXiv},
  author = {Daoguang Zan and Zhirong Huang and Wei Liu and Hanwu Chen and Shulin Xin and Linhao Zhang and Qi Liu and Aoyan Li and Lu Chen and Xiaojian Zhong and Siyao Liu and Yongsheng Xiao and Liangqiang Chen and Yuyu Zhang and Jing Su and Tianyu Liu and RUI LONG and Ming Ding and liang xiang},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  eprint = {2504.02605},
  title = {Multi-{SWE}-bench: A Multilingual Benchmark for Issue Resolving},
  url = {https://openreview.net/forum?id=MhBZzkz4h9},
  year = {2026},
}
""",
    )
