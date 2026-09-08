from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2503.07920"
_BIBTEX = r"""
@inproceedings{cahyawijaya2025seavl,
  title={Crowdsource, Crawl, or Generate? Creating {SEA}-{VL}, a Multicultural Vision-Language Dataset for Southeast Asia},
  author={Cahyawijaya, Samuel and Lovenia, Holy and Moniz, Joel Ruben Antony and Wong, Tack Hwa and Farhansyah, Mohammad Rifqi and Maung, Thant Thiri and Hudi, Frederikus and Anugraha, David and Habibi, Muhammad Ravi Shulthan and Qorib, Muhammad Reza and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18685--18717},
  year={2025}
}
"""
_DESCRIPTION = (
    "SEA-VL crawling is a large-scale, Southeast Asia–focused image–caption dataset, "
    "containing culturally relevant image–text pairs from the web. The subset used in MTEB "
    "features 2048 unique images and their corresponding captions, offering an evaluation "
    "benchmark for image–text and text–image retrieval in Southeast Asian cultural contexts."
)


class SeaVLCrawlingT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingT2IRetrieval",
        description=_DESCRIPTION
        + " Queries are captions; the corpus contains images (text→image retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/SEA-VL-Crawling-T2I",
            "revision": "761fb5ed934f053c8b94d321b7885cd5a3ad115f",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-03-10"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find an image that matches the given caption."},
        is_beta=True,
    )


class SeaVLCrawlingI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingI2TRetrieval",
        description=_DESCRIPTION
        + " Queries are images; the corpus contains captions (image→text retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/SEA-VL-Crawling-I2T",
            "revision": "2239cc6bc852b299bd5ecb3898d81d2ef29a17c0",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-03-10"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find a caption that matches the given image."},
        is_beta=True,
    )
