from __future__ import annotations

from hashlib import sha256

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks import AbsTaskAny2AnyRetrieval


class StanfordCarsI2I(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="StanfordCarsI2IRetrieval",
        description="Retrieve car images from 196 makes.",
        reference="https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content",
        dataset={
            "path": "isaacchung/StanfordCars",
            "revision": "09ffe9bc7864d3f1e851529e5c4b7e05601a04fb",
        },
        type="Retrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2012-01-01", "2013-04-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{Krause2013CollectingAL,
        title={Collecting a Large-scale Dataset of Fine-grained Cars},
        author={Jonathan Krause and Jia Deng and Michael Stark and Li Fei-Fei},
        year={2013},
        url={https://api.semanticscholar.org/CorpusID:16632981}
        }
        """,
        descriptive_stats={
            "n_samples": {"default": 8041},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1074.894348894349,
                    "average_query_length": 77.06142506142506,
                    "num_documents": 8041,
                    "num_queries": 8041,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        eval_split = self.metadata_dict["eval_splits"][0]
        data_raw = datasets.load_dataset(**self.metadata_dict["dataset"])[eval_split]

        queries = {eval_split: {"id": [], "modality": [], "image": []}}
        corpus = {eval_split: {"id": [], "modality": [], "image": []}}
        relevant_docs = {eval_split: {}}

        label_ids = {
            label: label
            for label in set(data_raw["label"])
        }

        for row in data_raw:
            image = row["image"]
            label = row["label"]
            query_id = row["id"]
            queries[eval_split]["id"].append(query_id)
            queries[eval_split]["image"].append(image)
            queries[eval_split]["modality"].append("image")

            doc_id = label_ids[label]
            corpus[eval_split]["id"].append(doc_id)
            corpus[eval_split]["image"].append(image)
            corpus[eval_split]["modality"].append("image")

            if query_id not in relevant_docs[eval_split]:
                relevant_docs[eval_split][query_id] = {}
            relevant_docs[eval_split][query_id][doc_id] = 1

        self.corpus = datasets.DatasetDict(corpus)
        self.queries = datasets.DatasetDict(queries)
        self.relevant_docs = datasets.DatasetDict(relevant_docs)

        self.data_loaded = True
