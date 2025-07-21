from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MusicCapsA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MusicCapsA2TRetrieval",
        description="Natural language description for music audio.",
        reference="https://github.com/nateraw/download-musiccaps-dataset",
        dataset={
            "path": "mteb/MusicCaps_a2t",
            "revision": "fb2fffc9a93f729d1844dfac3f8c16386fbe226bn",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Music"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{agostinelli2023musiclmgeneratingmusictext,
  archiveprefix = {arXiv},
  author = {Andrea Agostinelli and Timo I. Denk and Zalán Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matt Sharifi and Neil Zeghidour and Christian Frank},
  eprint = {2301.11325},
  primaryclass = {cs.SD},
  title = {MusicLM: Generating Music From Text},
  url = {https://arxiv.org/abs/2301.11325},
  year = {2023},
}
""",
    )


class MusicCapsT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MusicCapsT2ARetrieval",
        description="Natural language description for music audio.",
        reference="https://github.com/nateraw/download-musiccaps-dataset",
        dataset={
            "path": "mteb/MusicCaps_t2a",
            "revision": "995f536066b2bb7044f6b454f24a76a1c95caa33",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Music"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{agostinelli2023musiclmgeneratingmusictext,
  archiveprefix = {arXiv},
  author = {Andrea Agostinelli and Timo I. Denk and Zalán Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matt Sharifi and Neil Zeghidour and Christian Frank},
  eprint = {2301.11325},
  primaryclass = {cs.SD},
  title = {MusicLM: Generating Music From Text},
  url = {https://arxiv.org/abs/2301.11325},
  year = {2023},
}
""",
    )
