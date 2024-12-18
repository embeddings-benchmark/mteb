from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import islice
from typing import TypeVar

import datasets

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
            "path": "navjordj/SNL_summarization",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/navjordj/SNL_summarization",
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
        bibtex_citation="""@mastersthesis{navjord2023beyond,
  title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  year={2023},
  school={Norwegian University of Life Sciences, {\AA}s}
}""",
    )

    def dataset_transform(self):
        splits = self.metadata_dict["eval_splits"]

        documents: list = []
        labels: list = []
        label_col = "category"

        ds = {}
        for split in splits:
            ds_split = self.dataset[split]

            _label = self.normalize_labels(ds_split[label_col])
            documents.extend(ds_split["ingress"])
            labels.extend(_label)

            documents.extend(ds_split["article"])
            labels.extend(_label)

            assert len(documents) == len(labels)

            rng = random.Random(42)  # local only seed
            pairs = list(zip(documents, labels))
            rng.shuffle(pairs)
            documents, labels = (list(collection) for collection in zip(*pairs))

            # reduce size of dataset to not have too large datasets in the clustering task
            documents_batched = list(batched(documents, 512))[:4]
            labels_batched = list(batched(labels, 512))[:4]

            ds[split] = datasets.Dataset.from_dict(
                {
                    "sentences": documents_batched,
                    "labels": labels_batched,
                }
            )

        self.dataset = datasets.DatasetDict(ds)

    @staticmethod
    def normalize_labels(labels: list[str]) -> list[str]:
        # example label:
        # Store norske leksikon,Kunst og estetikk,Musikk,Klassisk musikk,Internasjonale dirigenter
        # When using 2 levels there is 17 unique labels
        # When using 3 levels there is 121 unique labels
        return [",".join(tuple(label.split(",")[:3])) for label in labels]
