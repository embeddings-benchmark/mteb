from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "Jun-Yang/OmniCVR"
_DATASET_REVISION = "81f254d1e5993dfec408fa111990150c32c3e50f"
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
    "instruction. The original benchmark evaluates within a per-query 2000-"
    "video gallery; this MTEB adaptation converts the task into standard "
    "global-corpus retrieval by taking the union of all candidate videos and "
    "deduplicating by video id, so every query is scored against the same "
    "shared corpus of ~16,316 videos. Because the corpus is ~8x larger than "
    "the paper's per-query gallery, absolute retrieval scores are not "
    "directly comparable to numbers reported in the original paper."
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

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Build a standard global-corpus retrieval split from OmniCVR."""
        if self.data_loaded:
            return

        from datasets import Video

        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        annotations = load_dataset(
            "json",
            data_files=f"https://huggingface.co/datasets/{path}/resolve/{revision}/omnicvr.jsonl",
            split="train",
        )
        videos = load_dataset(path, split="train", revision=revision)

        key_to_idx = {f"{k}.mp4": i for i, k in enumerate(videos["__key__"])}

        corpus_ids = sorted(
            {vid for cands in annotations["candidates"] for vid in cands}
        )
        corpus = (
            videos.select([key_to_idx[cid] for cid in corpus_ids])
            .add_column("id", corpus_ids)
            .rename_column("mp4", "video")
            .select_columns(["id", "video"])
            .cast_column("video", Video())
        )

        query_ids = [str(i) for i in range(len(annotations))]
        queries = (
            videos.select([key_to_idx[sid] for sid in annotations["source_id"]])
            .add_column("id", query_ids)
            .add_column("text", list(annotations["instruction"]))
            .rename_column("mp4", "video")
            .select_columns(["id", "video", "text"])
            .cast_column("video", Video())
        )

        qrels: dict[str, dict[str, int]] = {
            qid: {tid: 1} for qid, tid in zip(query_ids, annotations["target_id"])
        }

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=qrels,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True
