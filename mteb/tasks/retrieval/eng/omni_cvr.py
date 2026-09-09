from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/OmniCVR"
_DATASET_REVISION = "e0c1031c52fff76113b5917f05b1589ad3f0c61a"
_REFERENCE = "https://openreview.net/forum?id=KxxR7emO5K"
_BIBTEX = r"""
@inproceedings{ji2026omnicvr,
  author = {Junyang Ji and Shengjun Zhang and Da Li and Yuxiao Luo and Yan Wang and Di Xu and Biao Yang and Wei Yuan and Fan Yang and Zhihai He and Wenming Yang},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text},
  url = {https://openreview.net/forum?id=KxxR7emO5K},
  year = {2026},
}
"""
_DESCRIPTION = (
    "Composed video retrieval adapted from OmniCVR. Each query pairs a source "
    "video with a natural-language instruction describing a visual, acoustic, "
    "or integrated modification; the target is the video that satisfies the "
    "instruction. The shared corpus is the union of all candidate videos "
    "(~16,316 videos, deduplicated by video id), but evaluation preserves the "
    "original benchmark's per-query 2000-video gallery via the `top_ranked` "
    "candidate lists, so each query is only scored against its own gallery "
    "rather than the full shared corpus."
)


class OmniCVRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmniCVRVT2VRetrieval",
        description=_DESCRIPTION,
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-01-31"),
        domains=["Web", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the source video and the modification instruction, retrieve the video that satisfies the instruction."
        },
        is_beta=True,
    )
