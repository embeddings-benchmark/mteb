from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SoundDescsA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SoundDescsA2TRetrieval",
        description="Natural language description for different audio sources from the BBC Sound Effects webpage.",
        reference="https://github.com/akoepke/audio-retrieval-benchmark",
        dataset={
            "path": "mteb/audiocaps_a2t",
            "revision": "dbdd4928c401ff122c5b0d6c66accee653b3355c",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2022-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Natural Sound Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Koepke2022,
  author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
  booktitle = {IEEE Transactions on Multimedia},
  title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
  year = {2022},
}
""",
    )


class SoundDescsT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SoundDescsT2ARetrieval",
        description="Natural language description for different audio sources from the BBC Sound Effects webpage.",
        reference="https://github.com/akoepke/audio-retrieval-benchmark",
        dataset={
            "path": "mteb/sounddescs_t2a",
            "revision": "140da665f966e3871682813cf3d3030eb87d68bb",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2022-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Natural Sound Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Koepke2022,
  author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
  booktitle = {IEEE Transactions on Multimedia},
  title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
  year = {2022},
}
""",
    )
