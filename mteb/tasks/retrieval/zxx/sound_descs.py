from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SoundDescsA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SoundDescsA2TRetrieval",
        description="Natural language description for different audio sources from the BBC Sound Effects webpage.",
        reference="https://github.com/akoepke/audio-retrieval-benchmark",
        dataset={
            "path": "mteb/SoundDescsA2TRetrieval",
            "revision": "27f6d092e00721205296ebbb44f28ff2498d6b7e",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="hit_rate_at_5",
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


class SoundDescsT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SoundDescsT2ARetrieval",
        description="Natural language description for different audio sources from the BBC Sound Effects webpage.",
        reference="https://github.com/akoepke/audio-retrieval-benchmark",
        dataset={
            "path": "mteb/SoundDescsT2ARetrieval",
            "revision": "cfb2f05e4b75512e650468b1bf4d15891eec5be7",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="hit_rate_at_5",
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
