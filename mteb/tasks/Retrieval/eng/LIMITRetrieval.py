from collections.abc import Sequence

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_CITATION = """
@misc{weller2025theoreticallimit,
  archiveprefix = {arXiv},
  author = {Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
  eprint = {2508.21038},
  primaryclass = {cs.IR},
  title = {On the Theoretical Limitations of Embedding-Based Retrieval},
  url = {https://arxiv.org/abs/2508.21038},
  year = {2025},
}"""


class LIMITRetrieval(AbsTaskRetrieval):
    k_values: Sequence[int] = (1, 2, 3, 5, 10, 20, 100, 1000)
    metadata = TaskMetadata(
        name="LIMITRetrieval",
        description="A simple retrieval task designed to test all combinations of top-2 documents. This version includes all 50k docs.",
        reference="https://github.com/google-deepmind/limit",
        dataset={
            "path": "orionweller/LIMIT",
            "revision": "48142cc741b04d0b4af370ade7e8b42430382670",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_2",
        modalities=["text"],
        date=("2025-08-28", "2025-08-28"),
        domains=["Fiction"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
    )


class LIMITSmallRetrieval(AbsTaskRetrieval):
    k_values: Sequence[int] = (1, 2, 3, 5, 10, 20, 100, 1000)
    metadata = TaskMetadata(
        name="LIMITSmallRetrieval",
        description="A simple retrieval task designed to test all combinations of top-2 documents. This version only includes the 46 documents that are relevant to the 1000 queries.",
        reference="https://github.com/google-deepmind/limit",
        dataset={
            "path": "orionweller/LIMIT-small",
            "revision": "ff4a8f2476ae77476c1912f1f3cb5bb5f2d766d4",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_2",
        modalities=["text"],
        date=("2025-08-28", "2025-08-28"),
        domains=["Fiction"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
    )
