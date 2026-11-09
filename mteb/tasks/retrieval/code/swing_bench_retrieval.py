from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SwingBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SwingBenchRetrieval",
        description="A repository-level code retrieval task based on SwingBench, a set of real GitHub issues from Rust, Python, Go and C++ repositories, introduced with SwingArena. Each query is a GitHub issue description and is evaluated only against the files of its own repository snapshot (provided through the top_ranked subset), which together form the corpus. The relevant documents are the files modified by the reference solution.",
        reference="https://huggingface.co/datasets/SwingBench/SwingBench",
        dataset={
            "path": "mteb/SwingBenchRetrieval",
            "revision": "184a7b56751c3469154595198b6a8efeb615f231",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code", "rust-Code", "go-Code", "c++-Code"],
        main_score="ndcg_at_10",
        date=("2022-10-21", "2025-04-26"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a GitHub issue, retrieve the repository files that need to be modified to resolve the issue."
        },
        bibtex_citation=r"""
@misc{xu2025swingarena,
  archiveprefix = {arXiv},
  author = {Wendong Xu and Jing Xiong and Chenyang Zhao and Qiujiang Chen and Haoran Wang and Hui Shen and Zhongwei Wan and Jianbo Dai and Taiqiang Wu and He Xiao and Chaofan Tao and Z. Morley Mao and Ying Sheng and Zhijiang Guo and Hongxia Yang and Bei Yu and Lingpeng Kong and Quanquan Gu and Ngai Wong},
  eprint = {2505.23932},
  primaryclass = {cs.SE},
  title = {{SwingArena}: Competitive Programming Arena for Long-context {GitHub} Issue Solving},
  url = {https://arxiv.org/abs/2505.23932},
  year = {2025},
}
""",
    )
