from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VCDBCoreRetrieval(AbsTaskRetrieval):
    """Retrieve core videos with a human-annotated copied segment in common."""

    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="VCDBCoreRetrieval",
        description=(
            "Core-only video copy-detection retrieval derived from the VCDB partial "
            "copy detection dataset. All 528 core videos are both queries and corpus "
            "items; the remaining core videos act as implicit distractors for each "
            "query. The original temporal annotations are collapsed into 5,610 "
            "unique symmetric video-level relationships (11,220 directed qrels). "
            "The separate VCDB 100K-video background collection is not included."
        ),
        reference="https://doi.org/10.1007/978-3-319-10590-1_24",
        dataset={
            "path": "pranitchawla/VCDB-Core",
            "revision": "61bcc06abffe00cf1f6db50326bbba7f62094dc8",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="map_at_10",
        date=("2014-01-01", "2014-12-31"),
        domains=["Web"],
        task_subtypes=["Duplicate Detection"],
        license=(
            "https://web.archive.org/web/20251123150707/"
            "https://fvl.fudan.edu.cn/dataset/vcdb/list.htm"
        ),
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{jiang2014vcdb,
  author = {Jiang, Yu-Gang and Jiang, Yudong and Wang, Jiajun},
  booktitle = {Computer Vision -- ECCV 2014},
  pages = {357--371},
  publisher = {Springer},
  title = {VCDB: A Large-Scale Database for Partial Copy Detection in Videos},
  year = {2014},
}
""",
        prompt={
            "query": (
                "Retrieve another video that contains a copied segment in common "
                "with this video."
            )
        },
    )


class VCDBCoreAudioRetrieval(AbsTaskRetrieval):
    """Retrieve audio from core videos with a copied segment in common."""

    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="VCDBCoreAudioRetrieval",
        description=(
            "Audio-to-audio retrieval derived from the VCDB core video copy-detection "
            "dataset. Audio is available for 527 of the 528 core videos; the one "
            "video without an audio stream is excluded. The human temporal video "
            "annotations are collapsed into 5,584 unique symmetric relationships "
            "(11,168 directed qrels), so relevance is inherited from video-level "
            "copy annotations rather than independently judged for audio. The 527 "
            "tracks contain 461 unique audio files; repeated soundtracks from "
            "distinct source videos are retained as meaningful copy-detection items. "
            "The separate VCDB 100K-video background collection is not included."
        ),
        reference="https://doi.org/10.1007/978-3-319-10590-1_24",
        dataset={
            "path": "pranitchawla/VCDB-Core-Audio",
            "revision": "2304cf0464474d444982925012f1ea680ea77f03",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="map_at_10",
        date=("2014-01-01", "2014-12-31"),
        domains=["Web"],
        task_subtypes=["Duplicate Detection"],
        license=(
            "https://web.archive.org/web/20251123150707/"
            "https://fvl.fudan.edu.cn/dataset/vcdb/list.htm"
        ),
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{jiang2014vcdb,
  author = {Jiang, Yu-Gang and Jiang, Yudong and Wang, Jiajun},
  booktitle = {Computer Vision -- ECCV 2014},
  pages = {357--371},
  publisher = {Springer},
  title = {VCDB: A Large-Scale Database for Partial Copy Detection in Videos},
  year = {2014},
}
""",
        prompt={
            "query": (
                "Retrieve another audio track whose source video contains an "
                "annotated copied segment in common with this one."
            )
        },
    )
