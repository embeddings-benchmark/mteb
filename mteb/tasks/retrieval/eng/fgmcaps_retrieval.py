from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{figma2026,
  author = {Anand, Nishit and Seth, Ashish and Ghosh, Sreyan and Manocha, Dinesh and Duraiswami, Ramani},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  title = {FIGMA: Towards FIne-Grained Music retrievAl},
  url = {https://arxiv.org/abs/2606.06615},
  year = {2026},
}
"""


class FGMCapsT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FGMCapsT2ARetrieval",
        description=(
            "FGMCaps-Test is a fine-grained music retrieval benchmark containing "
            "10,000 10-second music clips. Each clip is paired with an English "
            "caption describing precise musical attributes such as tempo, key, "
            "chord progression, and time signature. Text-to-audio retrieval uses "
            "each caption to retrieve its corresponding music clip."
        ),
        reference="https://arxiv.org/abs/2606.06615",
        dataset={
            "path": "mteb/FGMCapsT2ARetrieval",
            "revision": "5502a16f85f7897d2d197c4402a5ff75f22c7e17",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2026-06-01", "2026-06-30"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the music clip described by this caption."},
    )


class FGMCapsA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FGMCapsA2TRetrieval",
        description=(
            "FGMCaps-Test is a fine-grained music retrieval benchmark containing "
            "10,000 10-second music clips. Each clip is paired with an English "
            "caption describing precise musical attributes such as tempo, key, "
            "chord progression, and time signature. Audio-to-text retrieval uses "
            "each music clip to retrieve its corresponding caption."
        ),
        reference="https://arxiv.org/abs/2606.06615",
        dataset={
            "path": "mteb/FGMCapsA2TRetrieval",
            "revision": "410bd3382b71088d28e62e9a95b2d21ecf5761f6",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2026-06-01", "2026-06-30"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the caption that describes this music clip."},
    )
