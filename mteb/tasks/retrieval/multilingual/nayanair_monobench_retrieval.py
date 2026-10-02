from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.types import RelevantDocumentsType

COMMON_METADATA = {
    "reference": "https://arxiv.org/abs/2512.03514",
    "type": "DocumentUnderstanding",
    "category": "t2i",
    "eval_splits": ["test"],
    "main_score": "ndcg_at_5",
    "date": ("2025-11-01", "2025-12-03"),
    "domains": ["Academic", "Engineering", "Financial", "Non-fiction"],
    "task_subtypes": ["Image Text Retrieval"],
    "license": "not specified",
    "annotations_creators": "LM-generated",
    "dialect": [],
    "modalities": ["text", "image"],
    "sample_creation": "multiple",
    "bibtex_citation": r"""@misc{kolavi2025m3druniversalmultilingualmultimodal,
  archiveprefix = {arXiv},
  author = {Adithya S Kolavi and Vyoman Jain},
  eprint = {2512.03514},
  primaryclass = {cs.CL},
  title = {M3DR: Towards Universal Multilingual Multimodal Document Retrieval},
  url = {https://arxiv.org/abs/2512.03514},
  year = {2025},
}""",
    "prompt": {"query": "Find a screenshot that is relevant to the user's question."},
}

_DESCRIPTION = (
    "Retrieve the document page image that answers a {lang} question. One of the 22 "
    "monolingual datasets of the Nayana-IR benchmark introduced with M3DR, distributed on "
    "the Hub as NayanaIR-MonoBench: ~1,000 document page images and ~200 queries per "
    "language, with graded relevance (2 = exact match, 1 = partial match elsewhere in the "
    "same document). The corpus is built by layout-aware translation of English source "
    "document images -- scientific papers, technical reports, educational materials, "
    "business documents and forms -- into the target language, and the queries are "
    "LM-generated, so this measures retrieval over synthetic rather than found documents."
)


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
) -> tuple[
    dict[str, Dataset],
    dict[str, Dataset],
    dict[str, RelevantDocumentsType],
]:
    corpus: dict[str, Dataset] = {}
    queries: dict[str, Dataset] = {}
    relevant_docs: dict[str, RelevantDocumentsType] = {}

    for split in splits:
        queries_ds = load_dataset(path, "queries", split=split, revision=revision)
        queries[split] = queries_ds.map(
            lambda x, split=split: {
                "id": f"query-{split}-{x['query-id']}",
                "text": x["query"],
                "modality": "text",
            },
            remove_columns=queries_ds.column_names,
        )

        corpus_ds = load_dataset(path, "corpus", split=split, revision=revision)
        corpus[split] = corpus_ds.map(
            lambda x, split=split: {
                "id": f"corpus-{split}-{x['corpus-id']}",
                "modality": "image",
            },
            remove_columns=["corpus-id", "doc-id"],
        )

        qrels_ds = load_dataset(path, "qrels", split=split, revision=revision)
        relevant_docs[split] = defaultdict(dict)
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"
            did = f"corpus-{split}-{row['corpus-id']}"
            relevant_docs[split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


def load_data(
    self: AbsTaskRetrieval, num_proc: int | None = None, **kwargs: Any
) -> None:
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = _load_data(
        path=self.metadata.dataset["path"],
        splits=self.metadata.eval_splits,
        revision=self.metadata.dataset["revision"],
    )

    self.data_loaded = True


class NayanaIRMonoBenchTamilRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchTamilRetrieval",
        description=_DESCRIPTION.format(lang="Tamil"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ta",
            "revision": "ee6bf80a8ef716b3c0d0206a6ac7044366929f71",
        },
        eval_langs=["tam-Taml"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchKannadaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchKannadaRetrieval",
        description=_DESCRIPTION.format(lang="Kannada"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-kn",
            "revision": "57aadad161dc544aae19a1940ad0725e5feb31a0",
        },
        eval_langs=["kan-Knda"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchGujaratiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchGujaratiRetrieval",
        description=_DESCRIPTION.format(lang="Gujarati"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-gu",
            "revision": "980d284e5f2e6c3ad0fe910d0702c661a047924a",
        },
        eval_langs=["guj-Gujr"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchPunjabiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchPunjabiRetrieval",
        description=_DESCRIPTION.format(lang="Punjabi"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-pa",
            "revision": "62b4a070e1c352c999bedb7d43886aa10936d7ce",
        },
        eval_langs=["pan-Guru"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchMalayalamRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchMalayalamRetrieval",
        description=_DESCRIPTION.format(lang="Malayalam"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ml",
            "revision": "c319524ed3b9dff43e9bba67f06b473fb75c7c3b",
        },
        eval_langs=["mal-Mlym"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchMarathiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchMarathiRetrieval",
        description=_DESCRIPTION.format(lang="Marathi"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-mr",
            "revision": "2be6a52d7ed244396595a884ea6ed8d9d67c55fb",
        },
        eval_langs=["mar-Deva"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchOdiaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchOdiaRetrieval",
        description=_DESCRIPTION.format(lang="Odia"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-or",
            "revision": "12e57c3efbc0abfeb4a3105886dfc5cefe9ecc89",
        },
        eval_langs=["ory-Orya"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBenchSanskritRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchSanskritRetrieval",
        description=_DESCRIPTION.format(lang="Sanskrit"),
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-sa",
            "revision": "a0c0cab1de4d8e25fd94a851f243b82f14487b8f",
        },
        eval_langs=["san-Deva"],
        **COMMON_METADATA,
    )

    load_data = load_data
