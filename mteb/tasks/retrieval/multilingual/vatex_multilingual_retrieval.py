from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/vatex-multilingual-retrieval"
_DATASET_REVISION = "8595f8bd6c29528c6d54f23e8640404465c459a2"

# VATEX annotates every clip in both English and Chinese, which is what makes a
# like-for-like cross-lingual comparison possible: identical videos, identical
# corpus, only the caption language changes.
_VATEX_LANGS = {
    "en": ["eng-Latn"],
    "zh": ["cmn-Hans"],
}

_BIBTEX = r"""
@inproceedings{wang2019vatex,
  author = {Wang, Xin and Wu, Jiawei and Chen, Junkun and Li, Lei and Wang, Yuan-Fang and Wang, William Yang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  title = {VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research},
  year = {2019},
}
"""

_DESCRIPTION = (
    "Cross-lingual video retrieval from VATEX, which captions each clip in both English "
    "and Chinese. Built from 993 validation clips carrying both languages, so the English "
    "and Chinese subsets share one video corpus and differ only in caption language. "
    "Distinct from the existing English VATEX tasks, which use the test split; Chinese "
    "captions are published only for validation, so the two do not overlap."
)


def _load_vatex_multilingual(task: AbsTaskRetrieval, direction: str) -> None:
    """Shared loader for both directions; the video corpus is fetched once."""
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    videos = load_dataset(
        _DATASET_PATH, "videos", revision=_DATASET_REVISION, split=split
    ).select_columns(["id", "video"])
    task.dataset = {}

    for lang in task.hf_subsets:
        caps = load_dataset(
            _DATASET_PATH, lang, revision=_DATASET_REVISION, split=split
        )
        text_ds = caps.select_columns(["id", "text"])

        links = caps.select_columns(["id", "video_id"]).to_dict()
        pairs = list(zip(links["id"], links["video_id"], strict=True))

        if direction == "t2v":
            queries, corpus = text_ds, videos
            qrels = {qid: {vid: 1} for qid, vid in pairs}
        else:
            corpus = text_ds
            qrels: dict[str, dict[str, int]] = {}
            for qid, vid in pairs:
                qrels.setdefault(vid, {})[qid] = 1
            keep = set(qrels)
            # select() by index rather than filter(), which would decode every video
            wanted = [i for i, id_ in enumerate(videos["id"]) if id_ in keep]
            queries = videos.select(wanted)

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=qrels,
                top_ranked=None,
            )
        }

    task.data_loaded = True


class VATEXMultilingualT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXMultilingualT2VRetrieval",
        description=(
            f"{_DESCRIPTION} Given a caption in English or Chinese, retrieve the video "
            "clip it describes."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=_VATEX_LANGS,
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="not specified",  # CC-BY-4.0 covers VATEX captions only, not the videos
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that matches the following caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex_multilingual(self, "t2v")


class VATEXMultilingualV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXMultilingualV2TRetrieval",
        description=(
            f"{_DESCRIPTION} Given a video clip, retrieve its English or Chinese captions."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=_VATEX_LANGS,
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="not specified",  # CC-BY-4.0 covers VATEX captions only, not the videos
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex_multilingual(self, "v2t")
