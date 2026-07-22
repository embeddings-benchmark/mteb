from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://huggingface.co/datasets/YqjMartin/FLARE"
_BIBTEX = r"""
@misc{flare2026,
  author = {YqjMartin},
  title = {FLARE: Full-Modality Long-Video Audiovisual Retrieval Benchmark with User-Simulated Queries},
  url = {https://huggingface.co/datasets/YqjMartin/FLARE},
  year = {2026},
}
"""
_DESCRIPTION = (
    "FLARE is a long-video audiovisual retrieval benchmark built from Video-MME "
    "source videos (399 videos, ~88k fine-grained clips). Clips have vision-only, "
    "audio-only, and unified audiovisual captions, plus user-simulated queries. "
    "The unified (cross-modal) query split applies a hard bimodal constraint so "
    "that only joint vision+audio evidence uniquely identifies the target.",
)


class FLAREUnifiedT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FLAREUnifiedT2VARetrieval",
        description=_DESCRIPTION
        + " This task uses unified user-simulated text queries to retrieve the "
        "matching audiovisual clip (text → video+audio). It is based on a 1k " 
        "subset of the original FLARE dataset.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/FLARE-1k-Unified-T2VA",
            "revision": "d9711a42e34a4ca1a3bf698252675e8941a5658f",
        },
        type="Any2AnyRetrieval",
        category="t2va",
        modalities=["text", "video", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2026-05-01"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval", "Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": (
                "Retrieve the video clip whose audiovisual content matches this "
                "description."
            )
        },
        is_beta=True,
    )


class FLAREVisionT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FLAREVisionT2VRetrieval",
        description=_DESCRIPTION
        + " This task uses vision-only user-simulated text queries to retrieve "
        "the matching video clip (text → video). It is based on a 1k subset of the "
        "original FLARE dataset.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/FLARE-1k-Vision-T2V",
            "revision": "acc178728cf51918d1674aed7da657f05fc068a7",
        },
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2026-05-01"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval", "Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the video clip that matches this visual description."
        },
        is_beta=True,
    )


class FLAREAudioT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FLAREAudioT2VARetrieval",
        description=_DESCRIPTION
        + " This task uses audio-only user-simulated text queries to retrieve "
        "the matching audiovisual clip (text → video+audio). It is based on a 1k "
        "subset of the original FLARE dataset.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/FLARE-1k-Audio-T2VA",
            "revision": "7cd512b99482265a07d58902a9e831f6962415a9",
        },
        type="Any2AnyRetrieval",
        category="t2va",
        modalities=["text", "video", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2026-05-01"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval", "Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": (
                "Retrieve the video clip whose audio content matches this description."
            )
        },
        is_beta=True,
    )
