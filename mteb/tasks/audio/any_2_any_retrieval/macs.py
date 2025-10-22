from __future__ import annotations

from mteb.abstasks.image.abs_task_any2any_retrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MACSA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MACSA2TRetrieval",
        description="Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.",
        reference="https://zenodo.org/records/5114771",
        dataset={
            "path": "mteb/MACS_a2t",
            "revision": "4b34d10b24e99551890d407f12732a2c2ee1405b",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{martinmorato2021groundtruthreliabilitymultiannotator,
  archiveprefix = {arXiv},
  author = {Irene Martin-Morato and Annamaria Mesaros},
  eprint = {2104.04214},
  primaryclass = {eess.AS},
  title = {What is the ground truth? Reliability of multi-annotator data for audio tagging},
  url = {https://arxiv.org/abs/2104.04214},
  year = {2021},
}
""",
    )


class MACST2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MACST2ARetrieval",
        description="Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.",
        reference="https://zenodo.org/records/5114771",
        dataset={
            "path": "mteb/MACS_t2a",
            "revision": "74ce553cb9b43aee5169f0bddd30bd5dee8d26d1",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{martinmorato2021groundtruthreliabilitymultiannotator,
  archiveprefix = {arXiv},
  author = {Irene Martin-Morato and Annamaria Mesaros},
  eprint = {2104.04214},
  primaryclass = {eess.AS},
  title = {What is the ground truth? Reliability of multi-annotator data for audio tagging},
  url = {https://arxiv.org/abs/2104.04214},
  year = {2021},
}
""",
    )
