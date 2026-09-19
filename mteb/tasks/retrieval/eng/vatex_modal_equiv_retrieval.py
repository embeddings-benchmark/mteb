"""VATEX Modality Equivalence retrieval tasks.

Six retrieval directions over a shared 150-clip pool built from mteb/VATEX_test_1k,
where every item is simultaneously available as a video clip, an audio track,
and an English text caption.

Because the candidate pool is IDENTICAL across all six directions, any difference
in retrieval score is attributable to modality difficulty rather than content.
This extends the COCO Modality Equivalence tasks (issue #5358) to the
{video, audio, text} triple.

Directions:
  v2t -- video -> text       (compare with a2t: which query modality is harder?)
  t2v -- text  -> video      (compare with t2a: which corpus modality is harder?)
  v2a -- video -> audio
  a2v -- audio -> video
  a2t -- audio -> text
  t2a -- text  -> audio
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/VATEX_test_1k"
_DATASET_REVISION = "0d2e86e6d36927f4676ee6127c4e38e3867ce0ce"
_POOL_SIZE = 150

_REFERENCE = "https://github.com/embeddings-benchmark/mteb/issues/5358"

_BIBTEX = r"""
@inproceedings{wang2019vatex,
  author = {Wang, Xin and Wu, Jiawei and Chen, Junkun and Li, Lei and Wang, Yuan-Fang and Wang, William Yang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  title = {VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research},
  year = {2019},
}
"""

_SHARED_POOL_NOTE = (
    "All six tasks in this group share the same 150-clip VATEX candidate pool "
    "where every item is simultaneously available as a video clip, audio track, "
    "and English text caption. Comparing scores across directions isolates the "
    "effect of modality from the effect of content. "
)

_COMMON = dict(
    reference=_REFERENCE,
    dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    type="Any2AnyRetrieval",
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("2019-01-01", "2019-12-31"),
    domains=["Activity", "Web"],
    task_subtypes=["Cross-Modal Retrieval"],
    license="not specified",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=_BIBTEX,
    is_beta=True,
)


def _load_vatex_modal_equiv(
    task: AbsTaskRetrieval,
    query_col: str,
    corpus_col: str,
) -> None:
    """Load a single modality-equivalence direction from the shared VATEX pool.

    Selects the first _POOL_SIZE items so all six directions use an identical
    candidate set. qrels are 1-to-1: each query maps to the corpus item that
    represents the same clip in the target modality.
    """
    if task.data_loaded:
        return

    dataset = load_dataset(
        task.metadata.dataset["path"],
        revision=task.metadata.dataset["revision"],
        split="test",
    )
    pool = dataset.select(range(min(_POOL_SIZE, len(dataset))))
    ids = [str(i) for i in range(len(pool))]
    pool = pool.add_column("id", ids)

    col_renames = {"caption": "text"}

    queries = pool.select_columns(["id", query_col])
    if query_col in col_renames:
        queries = queries.rename_column(query_col, col_renames[query_col])

    corpus = pool.select_columns(["id", corpus_col])
    if corpus_col in col_renames:
        corpus = corpus.rename_column(corpus_col, col_renames[corpus_col])

    qrels = {str(i): {str(i): 1} for i in range(len(pool))}

    task.dataset = {"default": {}}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class VATEXModalEquivV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivV2TRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are video clips; the corpus contains text captions. "
        "Compare with VATEXModalEquivA2TRetrieval to isolate the cost of video "
        "vs audio as the query modality when retrieving into text.",
        category="v2t",
        modalities=["video", "text"],
        prompt={"query": "Find the text caption that describes this video."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="video", corpus_col="caption")


class VATEXModalEquivT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivT2VRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are text captions; the corpus contains video clips. "
        "Compare with VATEXModalEquivT2ARetrieval to measure retrieval difficulty "
        "into a video corpus vs an audio corpus from the same text query.",
        category="t2v",
        modalities=["text", "video"],
        prompt={"query": "Find the video clip described by this caption."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="caption", corpus_col="video")


class VATEXModalEquivV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivV2ARetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are video clips; the corpus contains audio tracks. "
        "Tests cross-modal alignment between the visual and acoustic modalities "
        "of the same temporal content.",
        category="v2a",
        modalities=["video", "audio"],
        prompt={"query": "Find the audio track that corresponds to this video."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="video", corpus_col="audio")


class VATEXModalEquivA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivA2VRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are audio tracks; the corpus contains video clips. "
        "Tests cross-modal alignment between the acoustic and visual modalities "
        "of the same temporal content.",
        category="a2v",
        modalities=["audio", "video"],
        prompt={"query": "Find the video clip that corresponds to this audio."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="audio", corpus_col="video")


class VATEXModalEquivA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivA2TRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are audio tracks; the corpus contains text captions. "
        "Compare with VATEXModalEquivV2TRetrieval to isolate the cost of audio "
        "vs video as the query modality when retrieving into text.",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the text caption that describes this audio."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="audio", corpus_col="caption")


class VATEXModalEquivT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXModalEquivT2ARetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are text captions; the corpus contains audio tracks. "
        "Compare with VATEXModalEquivT2VRetrieval to measure retrieval difficulty "
        "into an audio corpus vs a video corpus from the same text query.",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the audio track described by this caption."},
        **_COMMON,
    )

    def load_data(self, **kwargs: Any) -> None:
        _load_vatex_modal_equiv(self, query_col="caption", corpus_col="audio")
