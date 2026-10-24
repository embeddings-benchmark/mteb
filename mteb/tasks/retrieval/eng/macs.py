from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MACSA2TRetrieval(AbsTaskRetrieval):
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
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        superseded_by="MACSA2TRetrieval.v2",
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


class MACST2ARetrieval(AbsTaskRetrieval):
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
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        superseded_by="MACST2ARetrieval.v2",
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


class MACSA2TRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MACSA2TRetrieval.v2",
        description="Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset."
        " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less.",
        reference="https://zenodo.org/records/5114771",
        dataset={
            "path": "mteb/MACS_a2t",
            "revision": "42039fff73458aac1e8e3919484485aa23d5be34",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["MACSA2TRetrieval"],
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


class MACST2ARetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MACST2ARetrieval.v2",
        description="Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset."
        " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less.",
        reference="https://zenodo.org/records/5114771",
        dataset={
            "path": "mteb/MACS_t2a",
            "revision": "8fc9588704d45687b0f36fd5baf772ce03ec93ac",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="https://zenodo.org/records/5114771",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["MACST2ARetrieval"],
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
