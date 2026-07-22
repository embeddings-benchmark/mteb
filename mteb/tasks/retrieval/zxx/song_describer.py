from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2311.10057"
_BIBTEX = r"""
@inproceedings{manco2023song,
  author = {Manco, Ilaria and Weck, Benno and Doh, Seungheon and Won, Minz and Zhang, Yixiao and Bogdanov, Dmitry and Wu, Yusong and Chen, Ke and Tovstogan, Philip and Benetos, Emmanouil and Quinton, Elio and Fazekas, Gy{\"o}rgy and Nam, Juhan},
  booktitle = {Machine Learning for Audio Workshop at NeurIPS 2023},
  title = {The Song Describer Dataset: a Corpus of Audio Captions for Music-and-Language Evaluation},
  url = {https://arxiv.org/abs/2311.10057},
  year = {2023},
}
"""
_DESCRIPTION = (
    "The Song Describer Dataset (SDD) is a corpus of human-written natural-language "
    "captions of music tracks, released for music-and-language evaluation. It "
    "provides 746 captions over 547 music recordings drawn from MTG-Jamendo "
    "(Creative Commons). "
)


class SongDescriberT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SongDescriberT2ARetrieval",
        description=_DESCRIPTION
        + "Text-to-music retrieval: the query is a caption and the goal is to "
        "retrieve the music track it describes from the corpus of 547 recordings.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SongDescriber-T2A",
            "revision": "f673048958",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-11-01"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the music recording described by this caption."},
    )


class SongDescriberA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SongDescriberA2TRetrieval",
        description=_DESCRIPTION
        + "Music-to-text retrieval: the query is a music recording and the goal is "
        "to retrieve its caption(s) from the corpus of 746 captions.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SongDescriber-A2T",
            "revision": "27e379ddf6",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-11-01"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the caption that describes this music recording."},
    )
