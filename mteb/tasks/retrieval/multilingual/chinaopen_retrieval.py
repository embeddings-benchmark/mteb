from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/ChinaOpen1k"
_DATASET_REVISION = "cd44dadbad44b5db820f50481bd6b3a42d5eeeb1"
_LANGUAGES = {
    "zho-Hans": ["zho-Hans"],
    "eng-Latn": ["eng-Latn"],
}

_CHINAOPEN_BIBTEX = r"""
@inproceedings{chen2023chinaopen,
  author = {Chen, Aozhu and Wang, Ziyuan and Dong, Chengbo and Tian, Kaibin and Zhao, Ruixiang and Liang, Xun and Kang, Zhanhui and Li, Xirong},
  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
  doi = {10.1145/3581783.3612156},
  title = {ChinaOpen: A Dataset for Open-world Multimodal Learning},
  year = {2023},
}
"""

_CHINAOPEN_DESCRIPTION_TAIL = (
    "Built from the manually annotated ChinaOpen-1k test set (1,092 Bilibili "
    "videos). The Chinese captions are the native annotation, written by human "
    "annotators watching the video, and the English captions are translations "
    "of them, so both language subsets describe the same videos and differ only "
    "in language. Uploader-written video titles are not used. Queries are "
    "deduplicated by caption text and every video carrying a caption is marked "
    "relevant, so the few captions shared by more than one video are "
    "multi-positive rather than incorrectly scored."
)


def _load_chinaopen(task: AbsTaskRetrieval, direction: str) -> None:
    """Shared loader for both ChinaOpen retrieval directions.

    The 1,092 videos live in one shared ``videos`` config, used as-is by both
    directions and both languages. Each language additionally has a
    ``<lang>-texts`` config (deduplicated captions) and a ``<lang>-links``
    config (every caption<->video pair, so a caption shared by more than one
    video produces several links). ``direction`` decides which side of each
    link becomes the query: ``t2v`` puts captions on the query side (so a
    shared caption is multi-positive), ``v2t`` puts videos on the query side
    (always exactly one correct caption per video).
    """
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    split = task.metadata.eval_splits[0]

    videos = load_dataset(path, "videos", split=split, revision=revision)

    task.dataset = {}
    for lang in task.hf_subsets:
        texts = load_dataset(path, f"{lang}-texts", split=split, revision=revision)
        links = load_dataset(path, f"{lang}-links", split=split, revision=revision)

        relevant_docs: dict[str, dict[str, int]] = {}
        for row in links:
            text_id, video_id = row["text-id"], row["video-id"]
            query_id, corpus_id = (
                (text_id, video_id) if direction == "t2v" else (video_id, text_id)
            )
            relevant_docs.setdefault(query_id, {})[corpus_id] = 1

        queries, corpus = (texts, videos) if direction == "t2v" else (videos, texts)

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
        }
    task.data_loaded = True


class ChinaOpenT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChinaOpenT2VRetrieval",
        description=(
            "Multilingual text-to-video retrieval over Chinese web video: given "
            "a caption in Chinese or English, retrieve the video it describes. "
            + _CHINAOPEN_DESCRIPTION_TAIL
        ),
        reference="https://ruc-aimc-lab.github.io/ChinaOpen/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyMultilingualRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2023-01-01", "2023-12-31"),
        domains=["Web", "Entertainment"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_CHINAOPEN_BIBTEX,
        prompt={"query": "Find the video that matches the given caption."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chinaopen(self, direction="t2v")


class ChinaOpenV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChinaOpenV2TRetrieval",
        description=(
            "Multilingual video-to-text retrieval over Chinese web video: given "
            "a video, retrieve the caption describing it from a corpus of "
            "Chinese or English captions. " + _CHINAOPEN_DESCRIPTION_TAIL
        ),
        reference="https://ruc-aimc-lab.github.io/ChinaOpen/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyMultilingualRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2023-01-01", "2023-12-31"),
        domains=["Web", "Entertainment"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_CHINAOPEN_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chinaopen(self, direction="v2t")
