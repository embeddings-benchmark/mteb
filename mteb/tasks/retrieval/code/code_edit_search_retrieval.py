from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = [
    "python",
    "javascript",
    "typescript",
    "go",
    "ruby",
    "java",
    "php",
    "c",
    "c++",
    "rust",
    "swift",
    "scala",
    "shell",
]


class CodeEditSearchRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "train"
    metadata = TaskMetadata(
        name="CodeEditSearchRetrieval",
        description="The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.",
        reference="https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer",
        dataset={
            "path": "mteb/CodeEditSearchRetrieval",
            "revision": "73c19e2fb77b6edc43b7354c817fb9417d9732fe",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2011-02-12", "2016-01-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{muennighoff2023octopack,
  archiveprefix = {arXiv},
  author = {Niklas Muennighoff and Qian Liu and Armel Randy Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro Von Werra and Shayne Longpre},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2308.07124},
  title = {OctoPack: Instruction Tuning Code Large Language Models},
  url = {https://openreview.net/forum?id=mw1PWNSWZP},
  year = {2024},
}
""",
    )
