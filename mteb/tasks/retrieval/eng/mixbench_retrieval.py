from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Value, load_dataset
from huggingface_hub import hf_hub_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

_DATASET_REPO = "mixed-modality-search/MixBench2026"
_DATASET_REVISION = "17a9e705b2346b118a63f163f10e47325f9e9ecc"
_QRELS_REPO = "mixed-modality-search/MixBench25"
_QRELS_REVISION = "88e3916036ea0bdb205f4da885d6e947a565c1a0"
_REFERENCE = "https://arxiv.org/abs/2507.19054"
_PROMPT = {
    "query": "Retrieve a relevant item that represents the query.",
}
_BIBTEX = r"""
@article{li2025closing,
  author = {Li, Binxu and Zhang, Yuhui and Wang, Xiaohan and Liang, Weixin and Schmidt, Ludwig and Yeung-Levy, Serena},
  journal = {arXiv preprint arXiv:2507.19054},
  title = {Closing the Modality Gap for Mixed Modality Search},
  url = {https://arxiv.org/abs/2507.19054},
  year = {2025},
}
"""
_COMMON_DESCRIPTION = (
    "MixBench evaluates retrieval from a heterogeneous corpus in which text-only, "
    "image-only, and image-and-text documents compete in one ranking. The document "
    "payloads come from the pinned MixBench2026 release. That release omitted its "
    "documented qrel split, so the task recovers the original judgments verbatim from "
    "the pinned MixBench25 qrels TSV. The benchmark dataset card declares MIT, but the "
    "included media originate from datasets with differing licenses; the aggregate "
    "media license is therefore recorded as not specified. "
)


def _normalize_rows(
    dataset: Dataset, *, drop_empty_modalities: bool = False
) -> Dataset:
    """Normalize IDs and drop modalities absent from the entire split."""
    dataset = dataset.cast_column("id", Value("string"))
    if not drop_empty_modalities:
        return dataset
    empty_columns = [
        modality
        for modality in ("text", "image", "audio", "video")
        if modality in dataset.column_names
        and all(value is None for value in dataset[modality])
    ]
    return dataset.remove_columns(empty_columns)


def _load_split(config: str, split: str) -> Dataset:
    """Load only the pinned Parquet file needed by the retrieval task."""
    parquet_path = hf_hub_download(
        repo_id=_DATASET_REPO,
        filename=f"{config}/{split}.parquet",
        repo_type="dataset",
        revision=_DATASET_REVISION,
    )
    return load_dataset("parquet", data_files=parquet_path, split="train")


def _load_qrels(config: str) -> dict[str, dict[str, int]]:
    qrels_path = Path(
        hf_hub_download(
            repo_id=_QRELS_REPO,
            filename=f"{config}/qrels/qrels.tsv",
            repo_type="dataset",
            revision=_QRELS_REVISION,
        )
    )
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels[str(row["query_id"])][str(row["corpus_id"])] = int(row["score"])
    return dict(qrels)


class _MixBenchBase(AbsTaskRetrieval):
    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        config = self.metadata.dataset["name"]
        queries = _normalize_rows(
            _load_split(config, "queries"), drop_empty_modalities=True
        )
        corpus = _normalize_rows(_load_split(config, "mixed_corpus"))

        self.dataset = {
            "default": {
                "test": {
                    "queries": queries,
                    "corpus": corpus,
                    "relevant_docs": _load_qrels(config),
                    "top_ranked": None,
                }
            }
        }
        self.data_loaded = True


class MixBenchMSCOCO(_MixBenchBase):
    metadata = TaskMetadata(
        name="MixBenchMSCOCO",
        description=_COMMON_DESCRIPTION
        + "This subset uses MSCOCO captions as text queries and MSCOCO-derived documents.",
        reference=_REFERENCE,
        dataset={
            "path": _DATASET_REPO,
            "revision": _DATASET_REVISION,
            "name": "MSCOCO",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-31"),
        domains=["Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt=_PROMPT,
    )


class MixBenchGoogleWIT(_MixBenchBase):
    metadata = TaskMetadata(
        name="MixBenchGoogleWIT",
        description=_COMMON_DESCRIPTION
        + "This subset uses Google WIT title-and-reference-description text queries and WIT-derived documents.",
        reference=_REFERENCE,
        dataset={
            "path": _DATASET_REPO,
            "revision": _DATASET_REVISION,
            "name": "Google_WIT",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt=_PROMPT,
    )


class MixBenchVisualNews(_MixBenchBase):
    metadata = TaskMetadata(
        name="MixBenchVisualNews",
        description=_COMMON_DESCRIPTION
        + "This subset uses VisualNews captions as text queries and VisualNews-derived documents.",
        reference=_REFERENCE,
        dataset={
            "path": _DATASET_REPO,
            "revision": _DATASET_REVISION,
            "name": "VisualNews",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-31"),
        domains=["News"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt=_PROMPT,
    )


class MixBenchOVEN(_MixBenchBase):
    metadata = TaskMetadata(
        name="MixBenchOVEN",
        description=_COMMON_DESCRIPTION
        + "This subset uses OVEN image-and-text questions and OVEN-derived documents.",
        reference=_REFERENCE,
        dataset={
            "path": _DATASET_REPO,
            "revision": _DATASET_REVISION,
            "name": "OVEN",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt=_PROMPT,
    )
