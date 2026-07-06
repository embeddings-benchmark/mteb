from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/MSR-VTT"
# Standard-format configs are produced by scripts/upload_msr_vtt_retrieval.py.
# Update this revision after reuploading the dataset.
_DATASET_REVISION = "4661603cee25c1fd370e5478a2953203cf37155b"
_BIBTEX = r"""
@inproceedings{xu2016msrvtt,
  author = {Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Msr-vtt: A large video description dataset for bridging video and language},
  year = {2016},
}
"""


class MSRVTTV2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2T",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTV2T",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="v2t",
        modalities=["video", "text"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )


class MSRVTTT2V(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTT2V",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTT2V",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="t2v",
        modalities=["text", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )


class MSRVTTVA2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTVA2T",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTVA2T",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="va2t",
        modalities=["audio", "video", "text"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )


class MSRVTTT2VA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTT2VA",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTT2VA",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="t2va",
        modalities=["text", "audio", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )


class MSRVTTV2A(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2A",
        description=(
            "Retrieve the audio track that matches a given video clip from MSR-VTT. "
            "Tests cross-modal alignment between video frames and audio."
            " Used the msrvtt_ret_test1k retrieval split (879 examples)."
        ),
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTV2A",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="v2a",
        modalities=["video", "audio"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )


class MSRVTTA2V(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTA2V",
        description=(
            "Retrieve the video clip that matches a given audio track from MSR-VTT. "
            "Tests cross-modal alignment between audio and video frames."
            " Used the msrvtt_ret_test1k retrieval split (879 examples)."
        ),
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTA2V",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="a2v",
        modalities=["audio", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )


class MSRVTTVT2A(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTVT2A",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTVT2A",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="vt2a",
        modalities=["video", "text", "audio"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the audio that corresponds to the following video and its description."
        },
        is_beta=True,
    )


class MSRVTTAT2V(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTAT2V",
        description="A large video description dataset for bridging video and language. Used the msrvtt_ret_test1k retrieval split (879 examples).",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "MSRVTTAT2V",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="at2v",
        modalities=["audio", "text", "video"],
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the video that corresponds to the following audio and its description."
        },
        is_beta=True,
    )
