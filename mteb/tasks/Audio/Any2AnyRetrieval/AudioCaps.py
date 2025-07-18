from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioCapsA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsA2TRetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "mteb/audiocaps_a2t",
            "revision": "9633a2d50398d042b79904bde006c1e23d7063d7",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
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


class AudioCapsT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsT2ARetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "mteb/audiocaps_t2a",
            "revision": "54610ade1c109ff008cd59f0f86578c3d6b9e330",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
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
