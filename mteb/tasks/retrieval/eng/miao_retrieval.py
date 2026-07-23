from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2605.17509"
_BIBTEX = r"""
@article{imoto2026audio,
  author = {Imoto, Keisuke and Kojima, Yamato and Tsuchiya, Takao},
  journal = {arXiv preprint arXiv:2605.17509},
  title = {Audio-Image Cross-Modal Retrieval with Onomatopoeic Images},
  year = {2026},
}
"""
_DESCRIPTION = (
    "MIAO (Multimodal Image-Audio Onomatopoeia) pairs CC0 FSD50K sound-event clips "
    "with human-drawn onomatopoeic illustrations (50 classes; 12 audio clips and 18 "
    "images per class). Relevance is class-level: an audio clip of class C is "
    "relevant to every image of class C, and vice versa. Media packaged via "
    "scripts/data/miao_retrieval/create_data.py from KeisukeImoto/MIAO."
)


class MIAOA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIAOA2IRetrieval",
        description=_DESCRIPTION
        + " Queries are sound-event clips and the corpus contains onomatopoeic "
        "images; the goal is to retrieve images depicting the same sound event.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/MIAO-A2I",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-05-17"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the onomatopoeic image that depicts this sound event."
        },
        is_beta=True,
    )


class MIAOI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIAOI2ARetrieval",
        description=_DESCRIPTION
        + " Queries are onomatopoeic images and the corpus contains sound-event "
        "clips; the goal is to retrieve audio of the depicted sound event.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/MIAO-I2A",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-05-17"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the sound-event clip depicted by this onomatopoeic image."
        },
        is_beta=True,
    )
