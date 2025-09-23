from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import TypeVar

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class SNLClustering(AbsTaskClustering):
    superseded_by = "SNLHierarchicalClusteringP2P"
    metadata = TaskMetadata(
        name="SNLClustering",
        dataset={
            "path": "mteb/SNLClustering",
            "revision": "e1c801d5a6fe26c89d5e878181246f5b292e6549",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/mteb/SNLClustering",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version is assumed (not specified before)
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
    )
