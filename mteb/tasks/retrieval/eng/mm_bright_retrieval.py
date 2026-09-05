from __future__ import annotations

import logging
from io import BytesIO
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
    from PIL import Image as PILImage

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

logger = logging.getLogger(__name__)

DATASET_PATH = "mm-bright/MM-BRIGHT"
DATASET_REVISION = "97702ca9ea81cd0a25288e74a9402439550d6bd4"
PAPER_REFERENCE = "https://arxiv.org/abs/2601.09562"

_DOMAINS = {
    "academia": ("Academia", "Academia"),
    "apple": ("Apple", "Apple"),
    "askubuntu": ("AskUbuntu", "Ask Ubuntu"),
    "aviation": ("Aviation", "Aviation"),
    "bioacoustics": ("Bioacoustics", "Bioacoustics"),
    "bioinformatics": ("Bioinformatics", "Bioinformatics"),
    "biology": ("Biology", "Biology"),
    "bitcoin": ("Bitcoin", "Bitcoin"),
    "chemistry": ("Chemistry", "Chemistry"),
    "christianity": ("Christianity", "Christianity"),
    "crypto": ("Crypto", "Cryptography"),
    "earthscience": ("EarthScience", "Earth Science"),
    "economics": ("Economics", "Economics"),
    "gaming": ("Gaming", "Gaming"),
    "gis": ("GIS", "GIS"),
    "islam": ("Islam", "Islam"),
    "law": ("Law", "Law"),
    "math": ("Math", "Mathematics"),
    "medicalsciences": ("MedicalSciences", "Medical Sciences"),
    "philosophy": ("Philosophy", "Philosophy"),
    "physics": ("Physics", "Physics"),
    "pm": ("ProjectManagement", "Project Management"),
    "psychology": ("Psychology", "Psychology"),
    "quant": ("Quant", "Quantitative Finance"),
    "quantumcomputing": ("QuantumComputing", "Quantum Computing"),
    "robotics": ("Robotics", "Robotics"),
    "salesforce": ("Salesforce", "Salesforce"),
    "sustainability": ("Sustainability", "Sustainability"),
    "travel": ("Travel", "Travel"),
}

_TaskVariant = Literal["t2t", "it2t", "it2i"]
_IMAGE_FEATURE = Image(mode="RGB")
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


def _concatenate_images_vertically(blobs: list[bytes]) -> PILImage.Image | None:
    from PIL import Image as PILImage

    images = []
    for blob in blobs:
        try:
            with PILImage.open(BytesIO(blob)) as image:
                images.append(image.convert("RGB"))
        except (OSError, ValueError):
            continue
    if not images:
        return None

    width = max(image.width for image in images)
    combined = PILImage.new(
        "RGB", (width, sum(image.height for image in images)), "white"
    )
    top = 0
    for image in images:
        combined.paste(image, ((width - image.width) // 2, top))
        top += image.height
    return combined


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
        image = _concatenate_images_vertically(
            [
                image_lookup[path]
                for path in example["image_paths"]
                if path in image_lookup
            ]
        )
        if image is None:
            from PIL import Image as PILImage

            missing_query_image_ids.add(example["id"])
            image = PILImage.new("RGB", (224, 224), "white")
        rows.append(
            {
                "id": example["id"],
                "text": example["query"],
                "image": image,
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


def _full_corpus_without_excluded_ids(
    documents: Dataset, examples: Dataset
) -> dict[str, list[str]]:
    document_ids = list(documents["id"])
    top_ranked = {}
    for example in examples:
        excluded_ids = set(example["negative_ids"])
        top_ranked[example["id"]] = (
            [
                document_id
                for document_id in document_ids
                if document_id not in excluded_ids
            ]
            if excluded_ids
            else document_ids
        )
    return top_ranked


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


def _filter_queries_without_relevant_images(
    queries: Dataset, qrels: dict, *, domain: str
) -> tuple[Dataset, dict]:
    query_ids = queries["id"]
    valid_indices = [
        index for index, query_id in enumerate(query_ids) if qrels[query_id]
    ]
    if len(valid_indices) != len(queries):
        logger.warning(
            "Dropping %d %s image-retrieval queries without a usable positive image",
            len(queries) - len(valid_indices),
            domain,
        )
    valid_ids = [query_ids[index] for index in valid_indices]
    return (
        queries.select(valid_indices),
        {query_id: qrels[query_id] for query_id in valid_ids},
    )


def _load_domain(domain: str, variant: _TaskVariant) -> RetrievalSplitData:
    config = "examples" if variant == "t2t" else "examples_multimodal"
    examples = _load_parquet(config, domain)

    if variant == "it2i":
        queries = _load_multimodal_queries(domain, examples)
        images = _load_image_data(domain, "document_images")
        qrels = _positive_image_qrels(examples, set(images["id"]))
        queries, qrels = _filter_queries_without_relevant_images(
            queries, qrels, domain=domain
        )
        return {
            "corpus": images,
            "queries": queries,
            "relevant_docs": qrels,
            # Task 3 negative_ids are text IDs, so they cannot exclude image IDs.
            "top_ranked": None,
        }

    documents = _load_documents(domain)
    if variant == "t2t":
        return {
            "corpus": documents,
            "queries": _text_queries(examples),
            "relevant_docs": _text_qrels(examples),
            "top_ranked": _full_corpus_without_excluded_ids(documents, examples),
        }

    queries = _load_multimodal_queries(domain, examples)
    return {
        "corpus": documents,
        "queries": queries,
        "relevant_docs": _text_qrels(examples),
        "top_ranked": _full_corpus_without_excluded_ids(documents, examples),
    }


def _load_mm_bright_domain(
    task: AbsTaskRetrieval,
    domain: str,
    variant: _TaskVariant,
    timer: TimingStack | None = None,
) -> None:
    if task.data_loaded:
        return
    timer = timer or TimingStack()
    with timer("Data loading", log_message=f"Loading dataset {task.metadata.name}..."):
        task.dataset = {"default": {"test": _load_domain(domain, variant)}}
    task.data_loaded = True


def _domain_metadata(domain: str, variant: _TaskVariant) -> TaskMetadata:
    class_name, display_name = _DOMAINS[domain]
    if variant == "t2t":
        suffix = "T2TRetrieval"
        task_type = "Retrieval"
        modalities = ["text"]
        task_subtypes = ["Reasoning as Retrieval"]
        description = (
            f"MM-BRIGHT text queries retrieving reasoning-intensive technical "
            f"passages in the {display_name} domain."
        )
        prompt = (
            "Given a technical question, retrieve passages that provide the "
            "reasoning needed to answer it."
        )
    elif variant == "it2t":
        suffix = "IT2TRetrieval"
        task_type = "Any2AnyRetrieval"
        modalities = ["text", "image"]
        task_subtypes = ["Reasoning as Retrieval", "Image Text Retrieval"]
        description = (
            f"MM-BRIGHT text-and-image queries retrieving reasoning-intensive "
            f"technical passages in the {display_name} domain."
        )
        prompt = (
            "Given a technical question and its images, retrieve passages that "
            "provide the reasoning needed to answer it."
        )
    else:
        suffix = "IT2IRetrieval"
        task_type = "Any2AnyRetrieval"
        modalities = ["text", "image"]
        task_subtypes = ["Reasoning as Retrieval", "Image Text Retrieval"]
        description = (
            f"MM-BRIGHT text-and-image queries retrieving relevant technical "
            f"images in the {display_name} domain."
        )
        prompt = (
            "Given a technical question and its images, retrieve images that "
            "provide relevant visual evidence."
        )

    return TaskMetadata(
        name=f"MMBright{class_name}{suffix}",
        description=description,
        type=task_type,
        category=variant,
        modalities=modalities,
        task_subtypes=task_subtypes,
        prompt={"query": prompt},
        eval_langs=["eng-Latn"],
        **_COMMON_METADATA,
    )


def _load_data(
    self: AbsTaskRetrieval,
    num_proc: int | None = None,
    *,
    timer: TimingStack | None = None,
    **kwargs: Any,
) -> None:
    _load_mm_bright_domain(self, self.domain, self.variant, timer=timer)


class MMBrightAcademiaT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "academia"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAcademiaIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "academia"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAcademiaIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "academia"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightAppleT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "apple"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAppleIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "apple"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAppleIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "apple"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightAskUbuntuT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "askubuntu"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAskUbuntuIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "askubuntu"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAskUbuntuIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "askubuntu"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightAviationT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "aviation"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAviationIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "aviation"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightAviationIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "aviation"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioacousticsT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioacoustics"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioacousticsIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioacoustics"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioacousticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioacoustics"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioinformaticsT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioinformatics"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioinformaticsIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioinformatics"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBioinformaticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bioinformatics"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightBiologyT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "biology"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBiologyIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "biology"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBiologyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "biology"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightBitcoinT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bitcoin"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBitcoinIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bitcoin"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightBitcoinIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "bitcoin"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightChemistryT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "chemistry"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightChemistryIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "chemistry"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightChemistryIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "chemistry"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightChristianityT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "christianity"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightChristianityIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "christianity"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightChristianityIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "christianity"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightCryptoT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "crypto"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightCryptoIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "crypto"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightCryptoIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "crypto"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightEarthScienceT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "earthscience"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightEarthScienceIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "earthscience"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightEarthScienceIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "earthscience"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightEconomicsT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "economics"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightEconomicsIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "economics"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightEconomicsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "economics"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightGamingT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gaming"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightGamingIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gaming"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightGamingIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gaming"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightGIST2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gis"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightGISIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gis"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightGISIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "gis"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightIslamT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "islam"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightIslamIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "islam"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightIslamIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "islam"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightLawT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "law"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightLawIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "law"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightLawIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "law"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightMathT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "math"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightMathIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "math"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightMathIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "math"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightMedicalSciencesT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "medicalsciences"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightMedicalSciencesIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "medicalsciences"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightMedicalSciencesIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "medicalsciences"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhilosophyT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "philosophy"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhilosophyIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "philosophy"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhilosophyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "philosophy"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhysicsT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "physics"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhysicsIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "physics"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPhysicsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "physics"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightProjectManagementT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "pm"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightProjectManagementIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "pm"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightProjectManagementIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "pm"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightPsychologyT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "psychology"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPsychologyIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "psychology"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightPsychologyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "psychology"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quant"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quant"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quant"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantumComputingT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quantumcomputing"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantumComputingIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quantumcomputing"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightQuantumComputingIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "quantumcomputing"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightRoboticsT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "robotics"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightRoboticsIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "robotics"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightRoboticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "robotics"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightSalesforceT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "salesforce"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightSalesforceIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "salesforce"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightSalesforceIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "salesforce"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightSustainabilityT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "sustainability"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightSustainabilityIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "sustainability"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightSustainabilityIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "sustainability"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)


class MMBrightTravelT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "travel"
    variant = "t2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightTravelIT2TRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "travel"
    variant = "it2t"
    metadata = _domain_metadata(domain, variant)


class MMBrightTravelIT2IRetrieval(AbsTaskRetrieval):
    load_data = _load_data
    domain = "travel"
    variant = "it2i"
    metadata = _domain_metadata(domain, variant)
