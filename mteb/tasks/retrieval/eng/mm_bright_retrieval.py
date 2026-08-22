from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
from datasets import (
    Dataset,
    DatasetInfo,
    Features,
    Image,
    Value,
)
from huggingface_hub import hf_hub_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.timing import TimingStack

if TYPE_CHECKING:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

logger = logging.getLogger(__name__)

DATASET_PATH = "mm-bright/MM-BRIGHT"
DATASET_REVISION = "97702ca9ea81cd0a25288e74a9402439550d6bd4"
PAPER_REFERENCE = "https://arxiv.org/abs/2601.09562"
PAIR_SEPARATOR = "|||"

DOMAINS = (
    "academia",
    "apple",
    "askubuntu",
    "aviation",
    "bioacoustics",
    "bioinformatics",
    "biology",
    "bitcoin",
    "chemistry",
    "christianity",
    "crypto",
    "earthscience",
    "economics",
    "gaming",
    "gis",
    "islam",
    "law",
    "math",
    "medicalsciences",
    "philosophy",
    "physics",
    "pm",
    "psychology",
    "quant",
    "quantumcomputing",
    "robotics",
    "salesforce",
    "sustainability",
    "travel",
)

_TaskVariant = Literal["t2t", "it2t", "it2i", "it2it"]
_IMAGE_FEATURE = Image(mode="RGB")
_PAIR_FEATURES = Features(
    {"id": Value("string"), "text": Value("string"), "image": _IMAGE_FEATURE}
)
_QUERY_FEATURES = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "image": _IMAGE_FEATURE,
    }
)

_COMMON_METADATA = dict(
    reference=PAPER_REFERENCE,
    dataset={"path": DATASET_PATH, "revision": DATASET_REVISION},
    eval_splits=["test"],
    eval_langs={domain: ["eng-Latn"] for domain in DOMAINS},
    main_score="ndcg_at_10",
    date=("2025-01-01", "2026-01-15"),
    domains=["Academic", "Web", "Medical", "Legal", "Religious"],
    license="cc-by-4.0",
    annotations_creators="expert-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@article{abdallah2026mmbright,
  archiveprefix = {arXiv},
  author = {Abdelrahman Abdallah and Mohamed Darwish Mounis and Mahmoud Abdalla and Mahmoud SalahEldin Kasem and Mostafa Farouk Senussi and Mohamed Mahmoud and Mohammed Ali and Adam Jatowt and Hyun-Soo Kang},
  eprint = {2601.09562},
  primaryclass = {cs.IR},
  title = {{MM-BRIGHT}: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval},
  url = {https://arxiv.org/abs/2601.09562},
  year = {2026},
}
""",
)


def _load_parquet(config: str, domain: str) -> Dataset:
    path = hf_hub_download(
        repo_id=DATASET_PATH,
        filename=f"{config}/{domain}.parquet",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return Dataset.from_parquet(path)


def _load_documents(domain: str) -> Dataset:
    return (
        _load_parquet("documents", domain)
        .select_columns(["id", "content"])
        .rename_column("content", "text")
    )


def _is_svg_payload(blob: bytes) -> bool:
    prefix = blob.lstrip()[:5].lower()
    return prefix.startswith((b"<svg", b"<?xml"))


def _load_image_data(domain: str, config: str) -> Dataset:
    raw = _load_parquet(config, domain).select_columns(["path", "bytes"])
    valid_indices = [
        index for index, blob in enumerate(raw["bytes"]) if not _is_svg_payload(blob)
    ]
    if len(valid_indices) != len(raw):
        logger.warning(
            "Dropping %d %s/%s SVG payloads that Pillow cannot decode",
            len(raw) - len(valid_indices),
            config,
            domain,
        )
    table = raw.data.table.take(pa.array(valid_indices))
    paths = table.column("path")
    blobs = table.column("bytes")
    if [len(chunk) for chunk in paths.chunks] != [len(chunk) for chunk in blobs.chunks]:
        raise ValueError(f"Misaligned image columns in {config}/{domain}")
    image_chunks = [
        pa.StructArray.from_arrays([blob_chunk, path_chunk], names=["bytes", "path"])
        for blob_chunk, path_chunk in zip(blobs.chunks, paths.chunks, strict=True)
    ]
    table = pa.table(
        {
            "id": paths,
            "image": pa.chunked_array(image_chunks, type=_IMAGE_FEATURE.pa_type),
        }
    )
    return Dataset(
        table,
        info=DatasetInfo(
            features=Features({"id": Value("string"), "image": _IMAGE_FEATURE})
        ),
    )


def _load_multimodal_queries(domain: str, examples: Dataset) -> Dataset:
    raw_images = _load_parquet("examples_images", domain).select_columns(
        ["path", "bytes"]
    )
    image_lookup = {
        path: blob
        for path, blob in zip(raw_images["path"], raw_images["bytes"], strict=True)
        if not _is_svg_payload(blob)
    }
    rows = []
    missing_query_image_ids = set()
    for example in examples:
        images = [
            {"path": path, "bytes": image_lookup[path]}
            for path in example["image_paths"]
            if path in image_lookup
        ]
        if not images:
            missing_query_image_ids.add(example["id"])
        rows.append(
            {
                "id": example["id"],
                "text": example["query"],
                "image": images[0] if images else None,
            }
        )
    if missing_query_image_ids:
        logger.warning(
            "%d %s queries have no usable stored query image",
            len(missing_query_image_ids),
            domain,
        )
    return Dataset.from_list(rows, features=_QUERY_FEATURES)


def _text_queries(examples: Dataset) -> Dataset:
    return Dataset.from_dict({"id": examples["id"], "text": examples["query"]})


def _text_qrels(examples: Dataset) -> dict:
    return {
        example["id"]: dict.fromkeys(example["gold_ids"], 1) for example in examples
    }


def _ordered_unique(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group))


def _document_candidates(examples: Dataset) -> dict[str, list[str]]:
    return {
        example["id"]: _ordered_unique(example["gold_ids"], example["negative_ids"])
        for example in examples
    }


def _select_candidates(
    dataset: Dataset,
    top_ranked: dict[str, list[str]],
    *,
    description: str,
) -> Dataset:
    dataset_ids = dataset["id"]
    candidate_ids = {
        candidate_id
        for query_candidates in top_ranked.values()
        for candidate_id in query_candidates
    }
    missing_ids = candidate_ids - set(dataset_ids)
    if missing_ids:
        first_missing = min(missing_ids)
        raise ValueError(
            f"{description} references {len(missing_ids)} missing candidates; "
            f"first missing ID: {first_missing}"
        )
    return dataset.select(
        [index for index, item_id in enumerate(dataset_ids) if item_id in candidate_ids]
    )


def _annotated_image_candidates(
    examples: Dataset,
    available_image_ids: set[str],
) -> dict[str, list[str]]:
    top_ranked = {}
    for example in examples:
        candidate_ids = _ordered_unique(
            [item["image_path"] for item in example["positive_images"]],
            [item["image_path"] for item in example["negative_images"]],
        )
        top_ranked[example["id"]] = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in available_image_ids
        ]
    return top_ranked


def _filter_evaluable_image_queries(
    queries: Dataset,
    top_ranked: dict[str, list[str]],
    qrels: dict,
    *,
    domain: str,
) -> tuple[Dataset, dict[str, list[str]], dict]:
    """Keep queries scored by the source evaluator, which omits empty qrels."""
    evaluable_ids = [
        query_id
        for query_id in queries["id"]
        if top_ranked[query_id] and qrels[query_id]
    ]
    if len(evaluable_ids) != len(queries):
        logger.warning(
            "Dropping %d %s image-reranking queries without a usable positive "
            "and candidate set",
            len(queries) - len(evaluable_ids),
            domain,
        )
    evaluable_id_set = set(evaluable_ids)
    query_indices = [
        index
        for index, query_id in enumerate(queries["id"])
        if query_id in evaluable_id_set
    ]
    return (
        queries.select(query_indices),
        {query_id: top_ranked[query_id] for query_id in evaluable_ids},
        {query_id: qrels[query_id] for query_id in evaluable_ids},
    )


def _document_base(document_id: str) -> str | None:
    parts = Path(document_id).stem.split("_")
    # Hard negatives use a 16-character hash rather than an image source key.
    if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{16}", parts[1]):
        return None
    if parts and re.fullmatch(r"[0-9a-f]{8}", parts[0]):
        return parts[0]
    return None


def _pair_id(document_id: str, image_id: str) -> str:
    return f"{document_id}{PAIR_SEPARATOR}{image_id}"


def _positive_image_qrels(
    examples: Dataset,
    available_image_ids: set[str],
) -> dict:
    qrels = {}
    for example in examples:
        qrels[example["id"]] = {
            item["image_path"]: 1
            for item in example["positive_images"]
            if item["image_path"] in available_image_ids
        }
    return qrels


def _matching_candidate_documents(
    source_passage_id: str,
    candidate_document_ids: list[str],
) -> list[str]:
    """Resolve an image's source to its released candidate passage chunk."""
    if source_passage_id in candidate_document_ids:
        return [source_passage_id]
    source_base = _document_base(source_passage_id)
    if source_base is None:
        return []
    return [
        document_id
        for document_id in candidate_document_ids
        if _document_base(document_id) == source_base
    ]


def _pair_reranking_data(
    documents: Dataset,
    images: Dataset,
    examples: Dataset,
) -> tuple[Dataset, dict, dict[str, list[str]]]:
    """Build Task 4 candidates with graded text and text-image relevance."""
    document_lookup = {
        document["id"]: document["text"]
        for document in documents.select_columns(["id", "text"])
    }
    image_column = images.data.column("image")
    image_lookup = {
        image_id: image_column[index].as_py()
        for index, image_id in enumerate(images["id"])
    }

    corpus_rows: dict[str, dict] = {}
    qrels = {}
    top_ranked = {}
    for example in examples:
        candidate_document_ids = _ordered_unique(
            example["gold_ids"], example["negative_ids"]
        )
        candidate_pair_ids = []
        relevant = {}

        for document_id in candidate_document_ids:
            corpus_rows.setdefault(
                document_id,
                {
                    "id": document_id,
                    "text": document_lookup[document_id],
                    "image": None,
                },
            )
            candidate_pair_ids.append(document_id)
            if document_id in example["gold_ids"]:
                relevant[document_id] = 1

        for field, relevance in (("positive_images", 2), ("negative_images", 0)):
            for item in example[field]:
                image_id = item["image_path"]
                image = image_lookup.get(image_id)
                if image is None:
                    continue
                matching_documents = _matching_candidate_documents(
                    item["source_passage_id"], candidate_document_ids
                )
                for document_id in matching_documents:
                    pair_id = _pair_id(document_id, image_id)
                    corpus_rows.setdefault(
                        pair_id,
                        {
                            "id": pair_id,
                            "text": document_lookup[document_id],
                            "image": image,
                        },
                    )
                    candidate_pair_ids.append(pair_id)
                    if relevance and document_id in example["gold_ids"]:
                        relevant[pair_id] = relevance

        qrels[example["id"]] = relevant
        top_ranked[example["id"]] = list(dict.fromkeys(candidate_pair_ids))

    return (
        Dataset.from_list(list(corpus_rows.values()), features=_PAIR_FEATURES),
        qrels,
        top_ranked,
    )


def _load_domain(domain: str, variant: _TaskVariant) -> RetrievalSplitData:
    documents = _load_documents(domain)
    config = "examples" if variant == "t2t" else "examples_multimodal"
    examples = _load_parquet(config, domain)

    if variant == "t2t":
        top_ranked = _document_candidates(examples)
        return {
            "corpus": _select_candidates(
                documents, top_ranked, description=f"{domain} text reranking"
            ),
            "queries": _text_queries(examples),
            "relevant_docs": _text_qrels(examples),
            "top_ranked": top_ranked,
        }

    queries = _load_multimodal_queries(domain, examples)
    if variant == "it2t":
        top_ranked = _document_candidates(examples)
        return {
            "corpus": _select_candidates(
                documents, top_ranked, description=f"{domain} multimodal reranking"
            ),
            "queries": queries,
            "relevant_docs": _text_qrels(examples),
            "top_ranked": top_ranked,
        }

    images = _load_image_data(domain, "document_images")
    if variant == "it2i":
        available_image_ids = set(images["id"])
        top_ranked = _annotated_image_candidates(examples, available_image_ids)
        qrels = _positive_image_qrels(examples, available_image_ids)
        queries, top_ranked, qrels = _filter_evaluable_image_queries(
            queries, top_ranked, qrels, domain=domain
        )
        return {
            "corpus": _select_candidates(
                images, top_ranked, description=f"{domain} image reranking"
            ),
            "queries": queries,
            "relevant_docs": qrels,
            "top_ranked": top_ranked,
        }

    document_top_ranked = _document_candidates(examples)
    candidate_documents = _select_candidates(
        documents,
        document_top_ranked,
        description=f"{domain} pair reranking",
    )
    pair_corpus, qrels, top_ranked = _pair_reranking_data(
        candidate_documents, images, examples
    )
    return {
        "corpus": pair_corpus,
        "queries": queries,
        "relevant_docs": qrels,
        "top_ranked": top_ranked,
    }


def _load_mm_bright_data(
    task: AbsTaskRetrieval,
    variant: _TaskVariant,
    timer: TimingStack | None = None,
) -> None:
    if task.data_loaded:
        return
    timer = timer or TimingStack()
    with timer("Data loading", log_message=f"Loading dataset {task.metadata.name}..."):
        task.dataset = {
            domain: {"test": _load_domain(domain, variant)}
            for domain in task.hf_subsets
        }
    task.data_loaded = True


class MMBrightT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMBrightT2TRetrieval",
        description="MM-BRIGHT text queries reranking annotated positive and hard-negative technical passages across 29 domains.",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_mm_bright_data(self, "t2t", timer=timer)


class MMBrightIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMBrightIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries reranking annotated positive and hard-negative technical passages across 29 domains.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_mm_bright_data(self, "it2t", timer=timer)


class MMBrightIT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMBrightIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries reranking annotated positive and negative technical images across 29 domains.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_mm_bright_data(self, "it2i", timer=timer)


class MMBrightIT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMBrightIT2ITRetrieval",
        description="MM-BRIGHT text-and-image queries reranking graded passage-and-image evidence against annotated hard negatives across 29 domains.",
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passage-and-image pairs that provide the reasoning needed to answer it."
        },
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_mm_bright_data(self, "it2it", timer=timer)
