from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class AudioCapsA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsA2TRetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "mteb/audiocaps_a2t",
            "revision": "acfbf827c27f81787800129443780c072dc8ae62",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "zxx-Zxxx"],
        main_score="hit_rate_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {Audiocaps: Generating captions for audios in the wild},
  year = {2019},
}
""",
        # prompt={"query": "Retrieve the answer to the question."},
    )


class AudioCapsT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsT2ARetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "mteb/audiocaps_t2a",
            "revision": "cb63d82bd4b2868f5e6410bf771d1c91bfc50203",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "zxx-Zxxx"],
        main_score="hit_rate_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {Audiocaps: Generating captions for audios in the wild},
  year = {2019},
}
""",
    )
