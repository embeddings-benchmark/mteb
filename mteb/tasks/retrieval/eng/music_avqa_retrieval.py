"""Cross-clip MUSIC-AVQA instrument retrieval tasks."""

from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2203.14072"
_BIBTEX = r"""
@article{li2022musicavqa,
  author = {Li, Bo and others},
  title = {Music Audio-Visual Question Answering},
  year = {2022},
  url = {https://arxiv.org/abs/2203.14072},
}
"""
_DESCRIPTION = (
    "Cross-clip music-instrument retrieval derived from the MUSIC-AVQA test split. "
    "For each of 22 instrument classes, five clips are queries and ten different "
    "clips are corpus items (110 queries, 220 corpus items, and 1,100 qrels per "
    "direction). Corpus relevance is shared instrument class. Query and corpus "
    "source video IDs are disjoint, so this measures instrument-level audio-video "
    "association rather than matching media from the same synchronous clip."
)


class MusicAVQAA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MusicAVQAA2VRetrieval",
        description=_DESCRIPTION
        + " The query is audio; retrieve videos of the same instrument.",
        reference=_REFERENCE,
        dataset={
            "path": "iamfortytwo/MusicAVQA-A2V-Retrieval",
            "revision": "fb70b8f6bbf02292a289a338cfbb31d9b8508f8f",
        },
        type="Any2AnyRetrieval",
        category="a2v",
        modalities=["audio", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Cross-Modal Retrieval", "Music Instrument Recognition"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve videos featuring the same instrument as this audio."
        },
    )


class MusicAVQAV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MusicAVQAV2ARetrieval",
        description=_DESCRIPTION
        + " The query is video; retrieve audio of the same instrument.",
        reference=_REFERENCE,
        dataset={
            "path": "iamfortytwo/MusicAVQA-V2A-Retrieval",
            "revision": "5d8f94b5738a0fd88b4241c866ad3ee9475a621c",
        },
        type="Any2AnyRetrieval",
        category="v2a",
        modalities=["video", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Cross-Modal Retrieval", "Music Instrument Recognition"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve audio featuring the same instrument as this video."},
    )
