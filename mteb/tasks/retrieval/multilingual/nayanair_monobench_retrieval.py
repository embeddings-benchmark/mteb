from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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


class NayanaIRMonoBenchTamilRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchTamilRetrieval",
        description=_DESCRIPTION.format(lang="Tamil"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchTamilRetrieval",
            "revision": "7563d4b065caed16f2d22aa1aa7bc4be4f01c0be",
        },
        eval_langs=["tam-Taml"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchKannadaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchKannadaRetrieval",
        description=_DESCRIPTION.format(lang="Kannada"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchKannadaRetrieval",
            "revision": "83c0fa56a4b2199b8095a2f8c1b7a842f0d74225",
        },
        eval_langs=["kan-Knda"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchGujaratiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchGujaratiRetrieval",
        description=_DESCRIPTION.format(lang="Gujarati"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchGujaratiRetrieval",
            "revision": "9b41c17a77c8bcea9b940705a3b62f7a9be46495",
        },
        eval_langs=["guj-Gujr"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchPunjabiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchPunjabiRetrieval",
        description=_DESCRIPTION.format(lang="Punjabi"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchPunjabiRetrieval",
            "revision": "3d8f2a3df3ffd8c6421df6cf78273736a692001a",
        },
        eval_langs=["pan-Guru"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchMalayalamRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchMalayalamRetrieval",
        description=_DESCRIPTION.format(lang="Malayalam"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchMalayalamRetrieval",
            "revision": "d46c3aeec5ec1d3ae2566d1b68a0fc0ddeb835fb",
        },
        eval_langs=["mal-Mlym"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchMarathiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchMarathiRetrieval",
        description=_DESCRIPTION.format(lang="Marathi"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchMarathiRetrieval",
            "revision": "abee2d8691230b6515becbf69084670f6ac4d2d9",
        },
        eval_langs=["mar-Deva"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchOdiaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchOdiaRetrieval",
        description=_DESCRIPTION.format(lang="Odia"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchOdiaRetrieval",
            "revision": "70b5671b4b81f79fae2cf07a9e8fd3dbc7b5a1f7",
        },
        eval_langs=["ory-Orya"],
        **COMMON_METADATA,
    )


class NayanaIRMonoBenchSanskritRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBenchSanskritRetrieval",
        description=_DESCRIPTION.format(lang="Sanskrit"),
        dataset={
            "path": "mteb/NayanaIRMonoBenchSanskritRetrieval",
            "revision": "094314400e6d11fb3695eacf8ae128fb1411d8bf",
        },
        eval_langs=["san-Deva"],
        **COMMON_METADATA,
    )
