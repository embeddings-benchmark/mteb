from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata



class PatentFnBClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PatentFnBClustering",
        description="this is modified AI-hub patent dataset for clustering evaluation. Domain : food & beverage patent. ",
        reference="https://huggingface.co/datasets/on-and-on/clustering_patent_FnB_manufacturing",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
        dataset={
            "path": "on-and-on/clustering_patent_FnB_manufacturing",
            "revision": "b3cad5e338fcb782b7607c2dcece61fba0072c4b",
        },
        date=("2017-01-01", "2020-01-01"),
        form="Written",
        domains=["Academic", "Engineering"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        # bibtex_citation= ... # removed for brevity
    )

    def dataset_transform(self):
        documents: list = []
        labels: list = []

        split = self.metadata.eval_splits[0]
        ds = {}

        documents.append(self.dataset[split]["sentences"])
        labels.append(self.dataset[split]["labels"])

        # # documents_batched = list(batched(documents, 512))
        # # labels_batched = list(batched(labels, 512))

        ds[split] = datasets.Dataset.from_dict(
            {
                "sentences": documents,
                "labels": labels,
            }
        )
        self.dataset = datasets.DatasetDict(ds)
