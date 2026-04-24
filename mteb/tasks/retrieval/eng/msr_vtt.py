from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/MSR-VTT"
_DATASET_REVISION = "4661603cee25c1fd370e5478a2953203cf37155b"
_BIBTEX = r"""
@inproceedings{xu2016msrvtt,
  author = {Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Msr-vtt: A large video description dataset for bridging video and language},
  year = {2016},
}
"""


def _load_msr_vtt(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all MSR-VTT retrieval directions.

    TODO: Reupload dataset in standard format and remove this custom load_data.
    """
    if task.data_loaded:
        return
    task.dataset = {"default": {}}
    dataset = load_dataset(
        task.metadata.dataset["path"],
        revision=task.metadata.dataset["revision"],
        split=task.metadata.eval_splits[0],
    )
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

    query = dataset.select_columns(["id"] + query_columns)
    corpus = dataset.select_columns(["id"] + corpus_columns)
    if "caption" in query_columns:
        query = query.rename_column("caption", "text")
    if "caption" in corpus_columns:
        corpus = corpus.rename_column("caption", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class MSRVTTV2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2T",
        description="A large video description dataset for bridging video and language",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="v2t",
        modalities=["video", "text"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(self, query_columns=["video"], corpus_columns=["caption"])


class MSRVTTT2V(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTT2V",
        description="A large video description dataset for bridging video and language",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="t2v",
        modalities=["text", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(self, query_columns=["caption"], corpus_columns=["video"])


class MSRVTTVA2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTVA2T",
        description="A large video description dataset for bridging video and language",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="va2t",
        modalities=["audio", "video", "text"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(
            self, query_columns=["video", "audio"], corpus_columns=["caption"]
        )


class MSRVTTT2VA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTT2VA",
        description="A large video description dataset for bridging video and language",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="t2va",
        modalities=["text", "audio", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(
            self, query_columns=["caption"], corpus_columns=["video", "audio"]
        )


class MSRVTTV2A(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2A",
        description=(
            "Retrieve the audio track that matches a given video clip from MSR-VTT. "
            "Tests cross-modal alignment between video frames and audio."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="v2a",
        modalities=["video", "audio"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(self, query_columns=["video"], corpus_columns=["audio"])


class MSRVTTA2V(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTA2V",
        description=(
            "Retrieve the video clip that matches a given audio track from MSR-VTT. "
            "Tests cross-modal alignment between audio and video frames."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="a2v",
        modalities=["audio", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_msr_vtt(self, query_columns=["audio"], corpus_columns=["video"])
