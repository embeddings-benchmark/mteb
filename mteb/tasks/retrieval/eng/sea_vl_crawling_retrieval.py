from __future__ import annotations

from datasets import Dataset, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_SOURCE = "SEACrowd/sea-vl_crawling"
_SOURCE_REVISION = "4723b4f00b68dc2fe649624628f3e8d50afa1e74"
_N_SAMPLES = 2048
_SHUFFLE_BUFFER = 10_000
_SEED = 42

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
    "SEA-VL crawling is a large-scale Southeast Asia–focused image–caption collection "
    "(~1.27M web-crawled culturally relevant pairs). For MTEB evaluation we deterministically "
    f"downsample to {_N_SAMPLES} image–caption pairs via a seeded streaming shuffle "
    f"(buffer={_SHUFFLE_BUFFER}), using the first non-empty caption per image."
)


def _first_caption(captions: list[str] | None) -> str | None:
    if not captions:
        return None
    for cap in captions:
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return None


def _sample_pairs(*, n_samples: int = _N_SAMPLES) -> list[dict]:
    """Stream-sample image–caption pairs without downloading the full 1.27M set."""
    stream = load_dataset(
        _SOURCE,
        revision=_SOURCE_REVISION,
        split="train",
        streaming=True,
    )
    stream = stream.shuffle(seed=_SEED, buffer_size=_SHUFFLE_BUFFER)

    pairs: list[dict] = []
    for row in stream:
        text = _first_caption(row.get("caption"))
        image = row.get("image")
        if text is None or image is None:
            continue
        pairs.append(
            {
                "id": str(row["id"]),
                "image": image,
                "text": text,
                "category": row.get("category"),
            }
        )
        if len(pairs) >= n_samples:
            break
    if len(pairs) < n_samples:
        raise RuntimeError(
            f"Only collected {len(pairs)} valid pairs from {_SOURCE}; "
            f"expected {n_samples}."
        )
    return pairs


def _build_t2i_split(pairs: list[dict]) -> RetrievalSplitData:
    query_rows = [
        {"id": f"q-{p['id']}", "text": p["text"], "modality": "text"} for p in pairs
    ]
    corpus_rows = [
        {"id": f"d-{p['id']}", "image": p["image"], "modality": "image"} for p in pairs
    ]
    relevant_docs = {f"q-{p['id']}": {f"d-{p['id']}": 1} for p in pairs}
    return RetrievalSplitData(
        queries=Dataset.from_list(query_rows),
        corpus=Dataset.from_list(corpus_rows).cast_column("image", Image()),
        relevant_docs=relevant_docs,
    )


def _build_i2t_split(pairs: list[dict]) -> RetrievalSplitData:
    query_rows = [
        {"id": f"q-{p['id']}", "image": p["image"], "modality": "image"} for p in pairs
    ]
    corpus_rows = [
        {"id": f"d-{p['id']}", "text": p["text"], "modality": "text"} for p in pairs
    ]
    relevant_docs = {f"q-{p['id']}": {f"d-{p['id']}": 1} for p in pairs}
    return RetrievalSplitData(
        queries=Dataset.from_list(query_rows).cast_column("image", Image()),
        corpus=Dataset.from_list(corpus_rows),
        relevant_docs=relevant_docs,
    )


class SeaVLCrawlingT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingT2IRetrieval",
        description=_DESCRIPTION
        + " Queries are captions; the corpus contains images (text→image retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": _SOURCE,
            "revision": _SOURCE_REVISION,
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

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        pairs = _sample_pairs()
        self.dataset = {"default": {"test": _build_t2i_split(pairs)}}
        self.data_loaded = True


class SeaVLCrawlingI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingI2TRetrieval",
        description=_DESCRIPTION
        + " Queries are images; the corpus contains captions (image→text retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": _SOURCE,
            "revision": _SOURCE_REVISION,
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

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        pairs = _sample_pairs()
        self.dataset = {"default": {"test": _build_i2t_split(pairs)}}
        self.data_loaded = True
