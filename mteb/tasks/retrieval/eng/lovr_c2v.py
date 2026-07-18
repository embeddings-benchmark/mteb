from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LoVRC2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LoVRC2VRetrieval",
        description=(
            "Clip-to-video retrieval from the LoVR long-video benchmark as "
            "packaged in UVRB: the query is a short clip and the corpus contains "
            "long videos; the goal is to retrieve the video the clip was taken "
            "from. This evaluation set samples 120 corpus videos with a fixed "
            "seed, re-encoded to 360p, together with all clip queries whose "
            "target is in the sample."
        ),
        reference="https://arxiv.org/abs/2510.27571",
        dataset={
            "path": "dukesun99/LoVR-C2V",
            "revision": "13a17a84ad1cf9b6a26f2a25367dd81899fda630",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-11-01"),
        domains=["Scene", "Entertainment"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{guo2025uvrb,
  archiveprefix = {arXiv},
  author = {Zhuoning Guo and Mingxin Li and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Xiaowen Chu},
  eprint = {2510.27571},
  primaryclass = {cs.CV},
  title = {Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum},
  url = {https://arxiv.org/abs/2510.27571},
  year = {2025},
}
""",
        prompt={"query": "Retrieve the long video that this clip was taken from."},
        is_beta=True,
    )
