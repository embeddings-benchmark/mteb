from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2108.00688"
_BIBTEX = r"""
@article{heidler2023soundingearth,
  author = {Heidler, Konrad and Mou, Lichao and Hu, Di and Jin, Pu and Li, Guangyao and Gan, Chuang and Wen, Ji-Rong and Zhu, Xiao Xiang},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  title = {Self-supervised audiovisual representation learning for remote sensing data},
  url = {https://arxiv.org/abs/2108.00688},
  volume = {116},
  year = {2023},
}
"""
_DESCRIPTION = (
    "SoundingEarth pairs geotagged field recordings from the Radio Aporee sound "
    "map with co-located aerial imagery; every location contributes exactly one "
    "image and one recording, giving instance-level cross-modal ground truth. "
    "This task uses 2,048 locations sampled with a fixed seed and 1,000 queries; recordings are "
    "repair-transcoded to mono FLAC capped at 120 seconds. "
)


class SoundingEarthA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SoundingEarthA2IRetrieval",
        description=_DESCRIPTION
        + "Queries are field recordings and the corpus contains the aerial images; "
        "the goal is to retrieve the image of the location where the recording was "
        "made.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SoundingEarth-A2I",
            "revision": "e97546c9ce77b71da1c8a97b6bb6c034f4378bee",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2006-01-01", "2021-10-01"),
        domains=["AudioScene", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the aerial image of the location where this field recording was made."
        },
    )


class SoundingEarthI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SoundingEarthI2ARetrieval",
        description=_DESCRIPTION
        + "Queries are aerial images and the corpus contains the field recordings; "
        "the goal is to retrieve the recording made at the depicted location.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SoundingEarth-I2A",
            "revision": "137f91ccf6a3ce892697bd6022560b22ac3342f8",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2006-01-01", "2021-10-01"),
        domains=["AudioScene", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the field recording made at the location shown in this aerial image."
        },
    )
