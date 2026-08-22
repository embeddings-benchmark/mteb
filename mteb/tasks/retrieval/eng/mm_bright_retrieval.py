from __future__ import annotations

import logging
import re
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from datasets import (
    Dataset,
    DatasetInfo,
    Features,
    Image,
    Value,
    concatenate_datasets,
)
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

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
NO_IMAGE = "__NO_IMAGE__"

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
_BLANK_IMAGE_BUFFER = BytesIO()
PILImage.new("RGB", (1, 1), "white").save(_BLANK_IMAGE_BUFFER, format="PNG")
_BLANK_IMAGE = {"bytes": _BLANK_IMAGE_BUFFER.getvalue(), "path": None}
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
    return prefix.startswith(b"<svg") or prefix.startswith(b"<?xml")


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
                "image": images[0] if images else _BLANK_IMAGE,
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


def _document_base(document_id: str) -> str | None:
    parts = Path(document_id).stem.split("_")
    # Hard negatives use a 16-character hash rather than an image source key.
    if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{16}", parts[1]):
        return None
    if parts and re.fullmatch(r"[0-9a-f]{8}", parts[0]):
        return parts[0]
    return None


def _image_base(image_path: str, domain: str) -> str | None:
    match = re.search(rf"{re.escape(domain)}_([0-9a-f]{{8}})_", image_path)
    return match.group(1) if match else None


def _pair_id(document_id: str, image_id: str) -> str:
    return f"{document_id}{PAIR_SEPARATOR}{image_id}"


def _generate_image_pairs(documents, images, pair_positions):
    image_storage = images.data.column("image")
    for image_index, document_index in pair_positions:
        document = documents[document_index]
        image_id = images[image_index]["id"]
        yield {
            "id": _pair_id(document["id"], image_id),
            "text": document["text"],
            "image": image_storage[image_index].as_py(),
        }


def _pair_corpus(
    domain: str, documents: Dataset, images: Dataset, examples: Dataset
) -> tuple[Dataset, set[str]]:
    document_ids = documents.data.column("id")
    document_texts = documents.data.column("text")
    no_image_ids = pc.binary_join_element_wise(
        document_ids, f"{PAIR_SEPARATOR}{NO_IMAGE}", ""
    )
    no_image_table = pa.table(
        {
            "id": no_image_ids,
            "text": document_texts,
            "image": pa.array(
                [_BLANK_IMAGE] * len(documents), type=_IMAGE_FEATURE.pa_type
            ),
        }
    )
    no_image_corpus = Dataset(no_image_table, info=DatasetInfo(features=_PAIR_FEATURES))

    image_ids = images["id"]
    image_sources: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        for field in ("positive_images", "negative_images"):
            for item in example[field]:
                base = _document_base(item["source_passage_id"])
                if base is not None:
                    image_sources[item["image_path"]].add(base)
    image_bases = [
        image_sources.get(image_id)
        or ({base} if (base := _image_base(image_id, domain)) is not None else set())
        for image_id in image_ids
    ]
    wanted_bases = set().union(*image_bases)
    document_positions: dict[str, list[int]] = defaultdict(list)
    for index, document_id in enumerate(documents["id"]):
        base = _document_base(document_id)
        if base in wanted_bases:
            document_positions[base].append(index)

    pair_positions = []
    represented_image_ids = set()
    for image_index, (image_id, bases) in enumerate(
        zip(image_ids, image_bases, strict=True)
    ):
        # An image can be associated with multiple source passages.
        for base in sorted(bases):
            for document_index in document_positions.get(base, []):
                pair_positions.append((image_index, document_index))
                represented_image_ids.add(image_id)

    image_pair_corpus = Dataset.from_generator(
        _generate_image_pairs,
        features=_PAIR_FEATURES,
        fingerprint=sha256(
            (
                "mm-bright-pairs-v2:"
                f"{domain}:{documents._fingerprint}:"
                f"{images._fingerprint}:{examples._fingerprint}"
            ).encode()
        ).hexdigest(),
        gen_kwargs={
            "documents": documents,
            "images": images,
            "pair_positions": pair_positions,
        },
    )
    return concatenate_datasets(
        [no_image_corpus, image_pair_corpus]
    ), represented_image_ids


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


def _graded_pair_qrels(
    examples: Dataset,
    represented_image_ids: set[str],
) -> dict:
    qrels = {}
    for example in examples:
        relevant = {
            _pair_id(document_id, NO_IMAGE): 1 for document_id in example["gold_ids"]
        }
        gold_by_base: dict[str, list[str]] = defaultdict(list)
        for document_id in example["gold_ids"]:
            base = _document_base(document_id)
            if base is not None:
                gold_by_base[base].append(document_id)
        for item in example["positive_images"]:
            image_id = item["image_path"]
            if image_id not in represented_image_ids:
                continue
            base = _document_base(item["source_passage_id"])
            if base is None:
                continue
            for document_id in gold_by_base.get(base, []):
                relevant[_pair_id(document_id, image_id)] = 2
        qrels[example["id"]] = relevant
    return qrels


def _load_domain(domain: str, variant: _TaskVariant) -> RetrievalSplitData:
    documents = _load_documents(domain)
    config = "examples" if variant == "t2t" else "examples_multimodal"
    examples = _load_parquet(config, domain)

    if variant == "t2t":
        return {
            "corpus": documents,
            "queries": _text_queries(examples),
            "relevant_docs": _text_qrels(examples),
            "top_ranked": None,
        }

    queries = _load_multimodal_queries(domain, examples)
    if variant == "it2t":
        return {
            "corpus": documents,
            "queries": queries,
            "relevant_docs": _text_qrels(examples),
            "top_ranked": None,
        }

    images = _load_image_data(domain, "document_images")
    if variant == "it2i":
        available_image_ids = set(images["id"])
        return {
            "corpus": images,
            "queries": queries,
            "relevant_docs": _positive_image_qrels(examples, available_image_ids),
            "top_ranked": None,
        }

    pair_corpus, represented_image_ids = _pair_corpus(
        domain, documents, images, examples
    )
    return {
        "corpus": pair_corpus,
        "queries": queries,
        "relevant_docs": _graded_pair_qrels(examples, represented_image_ids),
        "top_ranked": None,
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
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages across 29 domains.",
        type="Retrieval",
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
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages across 29 domains.",
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
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images across 29 domains.",
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
        description="MM-BRIGHT text-and-image queries retrieving graded passage-and-image evidence across 29 domains.",
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
