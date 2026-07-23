from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2501.07574"
_BIBTEX = r"""
@inproceedings{liu2025uncommon,
  author = {Liu, Xingchen and Tayal, Piyush and Wang, Jianyuan and Zarzar, Jesus and Monnier, Tom and Tertikas, Konstantinos and Duan, Jiali and Kleiman, Yanir and Neverova, Natalia and Vedaldi, Andrea and Novotny, David and Shapovalov, Roman},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {UnCommon Objects in 3D},
  url = {https://arxiv.org/abs/2501.07574},
  year = {2025},
}
"""
_DESC = (
    "Object-instance retrieval on uCO3D (UnCommon Objects in 3D), a large collection "
    "of 360-degree turntable videos of individual objects. Each 360-degree orbit is "
    "split into a viewpoint-disjoint first half and second half; the corpus is the "
    "second-half clip and the query comes from the first half, so matching requires "
    "viewpoint-invariant instance recognition rather than exact-view matching. 500 "
    "object instances sampled with a fixed seed; same-category instances are hard "
    "negatives. Videos re-encoded to 360p. "
)


class UCO3DI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCO3DI2VRetrieval",
        description=_DESC
        + "The query is a single frame; retrieve the video clip of the same object instance.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/uCO3D-I2V",
            "revision": "875932253634b5d2973a3741b4d0c47c227b27d4",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-01-01"),
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the video of the object shown in this image."},
    )


class UCO3DV2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCO3DV2VRetrieval",
        description=_DESC
        + "The query is the first-half orbit clip; retrieve the second-half clip of the same object instance.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/uCO3D-V2V",
            "revision": "b91417820f8d064a8e4dc10ffa2ebcfcd2090042",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-01-01"),
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the video of the same object shown in this video."},
    )


class UCO3DT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCO3DT2VRetrieval",
        description=_DESC
        + "The query is a short natural-language caption; retrieve the video clip of the described object instance.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/uCO3D-T2V",
            "revision": "2dcb2a8dec36ff7e020fc279114d3278c32b2f1a",
        },
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-01-01"),
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the video of the object described by this caption."},
    )
