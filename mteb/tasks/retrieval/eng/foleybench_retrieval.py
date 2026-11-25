from __future__ import annotations

from collections import defaultdict
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BASE_DATASET = {
    "path": "mteb/FoleyBench",
    "revision": "0398a0760b12bc3d142c0fc79ce100ee283c84d1",
}
_LONG_DATASET = {
    "path": "mteb/FoleyBenchLong",
    "revision": "635c0bf80f53e1619632d50b3fcd07ffb4c3a3cd",
}
_BIBTEX = r"""
@inproceedings{dixit2026foleybench,
  author = {Dixit, Satvik and Saito, Koichi and Zhong, Zhi and Mitsufuji, Yuki and Donahue, Chris},
  booktitle = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title = {FoleyBench: A Benchmark For Video-to-Audio Models},
  year = {2026},
}
"""
_COMMON = {
    "reference": "https://arxiv.org/abs/2511.13219",
    "type": "Any2AnyRetrieval",
    "eval_splits": ["test"],
    "main_score": "ndcg_at_10",
    "date": ("2025-01-01", "2025-11-17"),
    "domains": ["Scene", "AudioScene", "Web"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-nc-sa-4.0",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "is_beta": True,
    "bibtex_citation": _BIBTEX,
}
_BASE_DESCRIPTION = (
    "FoleyBench is a benchmark for Foley-style video-to-audio generation: ~10-second "
    "clips from FineVideo, V3C1 and LVBench, filtered so that the sound is non-speech, "
    "non-music and caused by something visible on screen. All 5,000 clips are used, "
    "each split into a silent video (re-encoded to at most 360 px high) and its audio "
    "track. "
)
_LONG_DESCRIPTION = (
    "FoleyBench-Long is the long-clip set of FoleyBench, a benchmark for Foley-style "
    "video-to-audio generation: 653 30-second clips from FineVideo and V3C1, filtered "
    "so that the sound is non-speech, non-music and caused by something visible on "
    "screen. All 653 clips are used, each split into a silent video (re-encoded to at "
    "most 360 px high) and its audio track. "
)
_V2A = "The query is a silent video; retrieve the sound recorded with it."
_VT2A = (
    "The query is a silent video with its FoleyBench caption, as in text-conditioned "
    "video-to-audio generation; retrieve the sound recorded with it."
)
_A2V = "The query is a sound; retrieve the silent video it was recorded with."
_V2A_PROMPT = {"query": "Retrieve the sound that matches this silent video."}
_VT2A_PROMPT = {
    "query": "Retrieve the sound that matches this silent video and its description."
}
_A2V_PROMPT = {"query": "Retrieve the silent video that matches this sound."}


class FoleyBenchV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchV2ARetrieval",
        description=_BASE_DESCRIPTION + _V2A,
        dataset=_BASE_DATASET,
        category="v2a",
        modalities=["video", "audio"],
        eval_langs=["zxx-Zxxx"],
        prompt=_V2A_PROMPT,
        **_COMMON,
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
        description=_BASE_DESCRIPTION + _VT2A,
        dataset=_BASE_DATASET,
        category="vt2a",
        modalities=["video", "text", "audio"],
        eval_langs=["eng-Latn"],
        prompt=_VT2A_PROMPT,
        **_COMMON,
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
        description=_BASE_DESCRIPTION + _A2V,
        dataset=_BASE_DATASET,
        category="a2v",
        modalities=["audio", "video"],
        eval_langs=["zxx-Zxxx"],
        prompt=_A2V_PROMPT,
        **_COMMON,
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


class FoleyBenchLongV2ARetrieval(FoleyBenchV2ARetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchLongV2ARetrieval",
        description=_LONG_DESCRIPTION + _V2A,
        dataset=_LONG_DATASET,
        category="v2a",
        modalities=["video", "audio"],
        eval_langs=["zxx-Zxxx"],
        prompt=_V2A_PROMPT,
        **_COMMON,
    )


class FoleyBenchLongVT2ARetrieval(FoleyBenchVT2ARetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchLongVT2ARetrieval",
        description=_LONG_DESCRIPTION + _VT2A,
        dataset=_LONG_DATASET,
        category="vt2a",
        modalities=["video", "text", "audio"],
        eval_langs=["eng-Latn"],
        prompt=_VT2A_PROMPT,
        **_COMMON,
    )


class FoleyBenchLongA2VRetrieval(FoleyBenchA2VRetrieval):
    metadata = TaskMetadata(
        name="FoleyBenchLongA2VRetrieval",
        description=_LONG_DESCRIPTION + _A2V,
        dataset=_LONG_DATASET,
        category="a2v",
        modalities=["audio", "video"],
        eval_langs=["zxx-Zxxx"],
        prompt=_A2V_PROMPT,
        **_COMMON,
    )
