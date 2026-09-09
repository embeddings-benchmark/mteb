from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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

_SAMPLING_NOTE = (
    "Uses a deterministic 500-query stratified subsample of the full "
    "5,000-query OmniCVR benchmark (seed 42; 100 audio-center / 114 "
    "visual-center / 286 integrated, proportional to the full category "
    "split of 1,000 / 1,141 / 2,859). See `scripts/data/omnicvr/create_data.py` "
    "for the exact, reproducible sampling code."
)


class OmniCVRMiniVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmniCVRMiniVT2VRetrieval",
        description=(
            "A downsampled version of OmniCVR composed video retrieval. Each "
            "query pairs a source video with a natural-language instruction "
            "describing a visual, acoustic, or integrated modification; the "
            "target is the video that satisfies the instruction. "
            + _SAMPLING_NOTE
            + " Each of the 500 queries keeps its original, unmodified "
            "2,000-video `top_ranked` gallery from the full benchmark; the "
            "shared corpus is the union of candidate videos actually "
            "referenced by those galleries (~14,400 videos, a pure dedup -- "
            "no candidate is dropped)."
        ),
        reference=_REFERENCE,
        dataset={
            "path": "myang333/OmniCVR-mini",
            "revision": "48d185f21b713980f963a910ccbdbb4f22f057c5",
        },
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
        adapted_from=["OmniCVRVT2VRetrieval"],
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the source video and the modification instruction, retrieve the video that satisfies the instruction."
        },
        is_beta=True,
    )


class OmniCVRMiniVT2VRetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmniCVRMiniVT2VRetrievalHardNegatives",
        description=(
            "A downsampled, hard-negative-mined version of OmniCVR composed "
            "video retrieval. Each query pairs a source video with a "
            "natural-language instruction describing a visual, acoustic, or "
            "integrated modification; the target is the video that satisfies "
            "the instruction. "
            + _SAMPLING_NOTE
            + " Unlike OmniCVRMiniVT2VRetrieval, each query's original "
            "2,000-video gallery is reduced to 250 candidates (249 hard "
            "negatives + the positive). As an implementation choice for "
            "applying microsoft/xclip-base-patch16 to video-only corpus "
            "sampling (XCLIP has no standard composed video+text query "
            "formulation, and is never given the source video or "
            "instruction text), candidates are ranked by XCLIP video-embedding "
            "cosine similarity to the positive/target video -- itself a "
            "corpus entry, not a query field -- and the 249 most similar are "
            "kept alongside the positive. The shared corpus is the union of "
            "candidates actually retained across all 500 reduced galleries "
            "(~13,700 videos)."
        ),
        reference=_REFERENCE,
        dataset={
            "path": "myang333/OmniCVR-mini-hard-negatives",
            "revision": "18372ec526d991ee262536f6e37a7db1eef1ef8d",
        },
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
        adapted_from=["OmniCVRVT2VRetrieval"],
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the source video and the modification instruction, retrieve the video that satisfies the instruction."
        },
        is_beta=True,
    )
