from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2005.08449"
_BIBTEX = r"""
@inproceedings{hu2020cross,
  author = {Hu, Di and Li, Xuhong and Mou, Lichao and Jin, Pu and Chen, Dong and Jing, Liping and Zhu, Xiaoxiang and Dou, Dejing},
  booktitle = {European Conference on Computer Vision},
  organization = {Springer},
  pages = {68--84},
  title = {Cross-task transfer for geotagged audiovisual aerial scene recognition},
  year = {2020},
}
"""
_DESCRIPTION = (
    "ADVANCE pairs geotagged field recordings sourced from FreeSound with "
    "co-located 512x512 aerial imagery extracted from Google Earth; every "
    "location contributes exactly one image and one recording, giving "
    "instance-level cross-modal ground truth across 5,075 locations spanning "
    "13 land-cover scene categories. Note: at original resolution this "
    "dataset is ~6.7GB. "
)


class ADVANCEA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ADVANCEA2IRetrieval",
        description=_DESCRIPTION
        + "Queries are field recordings and the corpus contains the aerial "
        "images; the goal is to retrieve the image of the location where "
        "the recording was made.",
        reference=_REFERENCE,
        dataset={
            "path": "yaswanth169/ADVANCE-A2I",
            "revision": "38ef2712ff91da1b9c63c14c373a88006dfc1e4f",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-08-23"),
        domains=["AudioScene", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the aerial image of the location where this field recording was made."
        },
    )


class ADVANCEI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ADVANCEI2ARetrieval",
        description=_DESCRIPTION
        + "Queries are aerial images and the corpus contains the field "
        "recordings; the goal is to retrieve the recording made at the "
        "depicted location.",
        reference=_REFERENCE,
        dataset={
            "path": "yaswanth169/ADVANCE-I2A",
            "revision": "487e3c58d14bc659a5d5a32b6f3eeaf75841f9aa",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-08-23"),
        domains=["AudioScene", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the field recording made at the location shown in this aerial image."
        },
    )
