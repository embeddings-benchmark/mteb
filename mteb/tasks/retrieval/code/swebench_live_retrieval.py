from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEBenchLiveRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEBenchLiveRetrieval",
        description="A repository-level code retrieval task based on SWE-bench-Live, a continuously updated set of real GitHub issues from Python repositories. Each query is a GitHub issue description and is evaluated only against the files of its own repository snapshot (provided through the top_ranked subset), which together form the corpus. The relevant documents are the files modified by the reference solution.",
        reference="https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live",
        dataset={
            "path": "mteb/SWEBenchLiveRetrieval",
            "revision": "07906be7a15b8b738a39cb8a54e2aa7022ba866b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-07-22", "2025-09-02"),
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
@misc{zhang2025swebenchlive,
  archiveprefix = {arXiv},
  author = {Linghao Zhang and Shilin He and Chaoyun Zhang and Yu Kang and Bowen Li and Chengxing Xie and Junhao Wang and Maoquan Wang and Yufan Huang and Shengyu Fu and Elsie Nallipogu and Qingwei Lin and Yingnong Dang and Saravan Rajmohan and Dongmei Zhang},
  eprint = {2505.23419},
  primaryclass = {cs.SE},
  title = {{SWE}-bench Goes Live!},
  url = {https://arxiv.org/abs/2505.23419},
  year = {2025},
}
""",
    )
