from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://andrewowens.com/vis/"
_BIBTEX = r"""
@inproceedings{owens2016visually,
  author = {Owens, Andrew and Isola, Phillip and McDermott, Josh and Torralba, Antonio and Adelson, Edward H. and Freeman, William T.},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Visually Indicated Sounds},
  year = {2016},
}
"""
_DESC = (
    "Audio-video retrieval on the Greatest Hits (Visually Indicated Sounds) dataset, "
    "in which a person strikes and scratches objects with a drumstick. Each item is a "
    "2-second clip centered on an annotated impact and labeled with the material struck "
    "(wood, metal, water, glass, ...). The task matches impacts across modalities by "
    "material. 992 impacts across 17 materials, sampled with a fixed seed. Videos "
    "re-encoded to 360p. "
)


class GreatestHitsA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreatestHitsA2VRetrieval",
        description=_DESC
        + "The query is an impact sound; retrieve the video clips of the same material.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/GreatestHits-A2V",
            "revision": "a9d0f5b7faf64c300d0c5e0a3f146ee48fae52da",
        },
        type="Any2AnyRetrieval",
        category="a2v",
        modalities=["audio", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2015-01-01", "2016-06-01"),
        domains=["Scene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve videos of the same material as this impact sound."},
    )


class GreatestHitsV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreatestHitsV2ARetrieval",
        description=_DESC
        + "The query is a video clip; retrieve the impact sounds of the same material.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/GreatestHits-V2A",
            "revision": "2c4fc4aab06afdd89d87747dc42b40884229e0b6",
        },
        type="Any2AnyRetrieval",
        category="v2a",
        modalities=["video", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2015-01-01", "2016-06-01"),
        domains=["Scene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve impact sounds of the same material as this video."},
    )
