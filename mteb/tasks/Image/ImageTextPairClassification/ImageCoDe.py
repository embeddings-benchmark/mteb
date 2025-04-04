from __future__ import annotations

import numpy as np
from datasets import DatasetDict, load_dataset

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ImageCoDe(AbsTaskImageTextPairClassification):
    texts_column_names = ["text"]
    images_column_names = [
        "correct_image",
        "false_image_1",
        "false_image_2",
        "false_image_3",
        "false_image_4",
        "false_image_5",
        "false_image_6",
        "false_image_7",
        "false_image_8",
        "false_image_9",
    ]

    metadata = TaskMetadata(
        name="ImageCoDe",
        description="Identify the correct image from a set of similar images based on a precise caption.",
        reference="https://aclanthology.org/2022.acl-long.241.pdf",
        dataset={
            "path": "JamieSJS/imagecode-multi",
            "revision": "d28adfd8b34fefa546fdf94bdc352622b2575f6c",
        },
        type="Compositionality",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="image_acc",
        date=("2022-05-22", "2022-05-27"),  # conference dates
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{krojer2022image,
  title={Image retrieval from contextual descriptions},
  author={Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
  journal={arXiv preprint arXiv:2203.15867},
  year={2022}
}
""",
        descriptive_stats={
            "n_samples": {"test": 25322},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 23020,
                    "num_queries": 2302,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        corpus = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "corpus",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )["corpus"]
        query = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "query",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )["test"]
        qrels = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "qrels",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )["test"]

        corpus_ids = corpus["id"]
        corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

        def build_mappings(qrels):
            correct_answers = {}
            candidate_pools = {}
            for row in qrels:
                qid = row["query-id"]
                cid = row["corpus-id"]
                if qid not in candidate_pools:
                    candidate_pools[qid] = []
                candidate_pools[qid].append(cid)
                if row["score"] == 1:
                    correct_answers[qid] = cid
            return correct_answers, candidate_pools

        correct_answers, candidate_pools = build_mappings(qrels)

        def process_example(example):
            qid = example["id"]
            correct_id = correct_answers[qid]
            candidates = candidate_pools[qid]

            candidate_indices = [corpus_id_to_idx[cid] for cid in candidates]

            correct_mask = np.array([cid == correct_id for cid in candidates])
            correct_idx = candidate_indices[np.argmax(correct_mask)]
            incorrect_indices = [
                idx for idx, mask in zip(candidate_indices, ~correct_mask) if mask
            ]
            return {
                "text": example["text"],
                "correct_image": corpus[correct_idx]["image"],
                **{
                    f"false_image_{i + 1}": corpus[idx]["image"]
                    for i, idx in enumerate(incorrect_indices[:9])
                },
            }

        dataset = query.map(
            process_example, num_proc=4, remove_columns=query.column_names
        )

        self.dataset = DatasetDict({"test": dataset})
        self.dataset_transform()
        self.data_loaded = True
