from __future__ import annotations

from collections import defaultdict
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/FoleyBench"
_DATASET_REVISION = "6b1d330111e85cbd637bc31b1a89534ebc2639e3"
_REFERENCE = "https://arxiv.org/abs/2511.13219"
_BIBTEX = r"""
@inproceedings{dixit2026foleybench,
  author = {Dixit, Satvik and Saito, Koichi and Zhong, Zhi and Mitsufuji, Yuki and Donahue, Chris},
  booktitle = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title = {FoleyBench: A Benchmark For Video-to-Audio Models},
  year = {2026},
}
"""
_DESCRIPTION = (
    "FoleyBench is a benchmark for Foley-style video-to-audio generation: ~10-second "
    "clips from FineVideo, V3C1 and LVBench, filtered so that the sound is non-speech, "
    "non-music and caused by something visible on screen. 1,000 of its 5,000 clips are "
    "sampled with a fixed seed, and each clip is split into a silent video (re-encoded "
    "to at most 360 px high) and its audio track. "
)


class FoleyBenchV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchV2ARetrieval",
        description=_DESCRIPTION
        + "The query is a silent video; retrieve the sound recorded with it.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        modalities=["video", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-11-17"),
        domains=["Scene", "AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the sound that matches this silent video."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["queries"] = split_data["queries"].select_columns(
                    ["id", "video"]
                )


class FoleyBenchVT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchVT2ARetrieval",
        description=_DESCRIPTION
        + "The query is a silent video with its FoleyBench caption, as in "
        "text-conditioned video-to-audio generation; retrieve the sound recorded "
        "with it.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2a",
        modalities=["video", "text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-11-17"),
        domains=["Scene", "AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the sound that matches this silent video and its description."
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["queries"] = split_data["queries"].select_columns(
                    ["id", "video", "text"]
                )


class FoleyBenchA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchA2VRetrieval",
        description=_DESCRIPTION
        + "The query is a sound; retrieve the silent video it was recorded with.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        modalities=["audio", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-11-17"),
        domains=["Scene", "AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the silent video that matches this sound."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Reuse the V2A data with roles swapped: sounds become queries, videos the corpus."""
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["queries"], split_data["corpus"] = (
                    split_data["corpus"],
                    split_data["queries"].select_columns(["id", "video"]),
                )
                relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
                for query_id, docs in split_data["relevant_docs"].items():
                    for doc_id, score in docs.items():
                        relevant_docs[doc_id][query_id] = score
                split_data["relevant_docs"] = dict(relevant_docs)
