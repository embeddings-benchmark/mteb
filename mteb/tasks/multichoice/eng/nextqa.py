from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, Video, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_SCRATCH_DIR = Path(os.environ.get("MTEB_VIDEO_SCRATCH", "/tmp/mteb_video_cache")) / "nextqa"


def _extract_video(raw: dict, qid: str) -> str:
    """Write raw video bytes to scratch dir; return absolute path.

    Skips if target already exists with matching size (idempotent across runs).
    """
    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(raw.get("path") or "video.mp4").suffix or ".mp4"
    target = _SCRATCH_DIR / f"{qid}{suffix}"
    data = raw["bytes"]
    if target.exists() and target.stat().st_size == len(data):
        return str(target)
    target.write_bytes(data)
    return str(target)


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
):
    corpus: dict[str, Dataset] = {}
    queries: dict[str, Dataset] = {}
    relevant_docs: dict[str, dict[str, dict[str, int]]] = {}
    top_ranked: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for split in splits:
        ds = load_dataset(path, revision=revision, split=split)
        ds_raw_video = ds.cast_column("video", Video(decode=False))

        corpus_rows: list[dict] = []
        query_rows: list[dict] = []
        relevant_docs[split] = {}

        for idx, row in enumerate(ds_raw_video):
            qid = f"q{idx}"
            video_path = _extract_video(row["video"], qid)
            query_rows.append(
                {
                    "id": qid,
                    "text": row["question"],
                    "video": {"path": video_path, "bytes": None},
                }
            )
            answer = row["answer"]
            relevant_docs[split][qid] = {}
            for j, candidate in enumerate(row["candidates"]):
                doc_id = f"{qid}_c{j}"
                corpus_rows.append({"id": doc_id, "text": candidate})
                top_ranked[split][qid].append(doc_id)
                relevant_docs[split][qid][doc_id] = 1 if candidate == answer else 0

        query_feats = Features(
            {
                "id": Value("string"),
                "text": Value("string"),
                "video": Video(decode=True),
            }
        )
        queries[split] = Dataset.from_list(query_rows, features=query_feats)
        corpus[split] = Dataset.from_list(corpus_rows)

    top_ranked_plain = {s: dict(d) for s, d in top_ranked.items()}
    return corpus, queries, relevant_docs, top_ranked_plain


class NExTQAVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NExTQAVideoCentricQA",
        description="NExT-QA is a video question answering benchmark targeting causal and temporal reasoning over everyday videos. Each example pairs a short video with a natural language question and 5 candidate answers, of which exactly one is correct. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate from its 5 choices.",
        reference="https://arxiv.org/abs/2105.08276",
        dataset={
            "path": "mteb/NExT-QA",
            "revision": "18efe467c7dfd207d3e1cd6642d5ba3b31e8b25d",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-05-18", "2021-05-18"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{xiao2021next,
  author = {Xiao, Junbin and Shang, Xindi and Yao, Angela and Chua, Tat-Seng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {9777-9786},
  title = {NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
  year = {2021},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
