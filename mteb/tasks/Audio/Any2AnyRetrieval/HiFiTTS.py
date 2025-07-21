from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class HiFiTTSA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="HiFiTTSA2TRetrieval",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_a2t",
            "revision": "4535eb399017ce24ce41cfcdc7c157bf76bcfa27",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )


class HiFiTTST2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Reverse text-to-audio retrieval for the Hi‑Fi TTS sentence–audio pairs."""

    metadata = TaskMetadata(
        name="HiFiTTST2ARetrieval",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_t2a",
            "revision": "c5783d12999ec5f6a9f337990cfd46cc131e2b0f",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )
