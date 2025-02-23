from __future__ import annotations

import logging
from collections import defaultdict

import datasets
from datasets import Dataset

from .abs_text_retrieval import AbsTextRetrieval

logger = logging.getLogger(__name__)

OLD_FORMAT_RERANKING_TASKS = [
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "WebLINXCandidatesReranking",
    "AlloprofReranking",
    "SyntecReranking",
    "VoyageMMarcoReranking",
    "ESCIReranking",
    "MIRACLReranking",
    "WikipediaRerankingMultilingual",
    "RuBQReranking",
    "T2Reranking",
    "MMarcoReranking",
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "NamaaMrTydiReranking",
]


class AbsTextReranking(AbsTextRetrieval):
    """Abstract class for re-ranking experiments. This is mostly the same as the RetrievalEvaluator, but here to adapt the old format to the new format. TODO: update these tasks to the new format and delete this class."""

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if self.metadata.name in OLD_FORMAT_RERANKING_TASKS:
            self.transform_old_dataset_format()
        else:
            # use AbsTaskRetrieval default to load the data
            return super().load_data(**kwargs)

    def process_example(self, example: dict, split: str, query_idx: int) -> dict:
        """Process a single example from the dataset."""
        query = example["query"]
        positive_docs = example["positive"]
        negative_docs = example["negative"]

        query_id = f"{split}_query{query_idx}"

        # Initialize the structures for this example
        example_data = {
            "query_id": query_id,
            "query": query,
            "doc_ids": [],
            "doc_texts": [],
            "relevance_scores": [],
        }

        for i, pos_doc in enumerate(positive_docs):
            # format i as a five digit number
            formatted_i = str(i).zfill(5)
            # have "a" in front so that positives are first, then negatives
            #   this shouldn't matter except for ties, and the previous reranking results
            #   had the positives first
            doc_id = f"apositive_{query_id}_{formatted_i}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(pos_doc)
            example_data["relevance_scores"].append(1)

        for i, neg_doc in enumerate(negative_docs):
            formatted_i = str(i).zfill(5)
            doc_id = f"negative_{query_id}_{formatted_i}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(neg_doc)
            example_data["relevance_scores"].append(0)

        return example_data

    def transform_old_dataset_format(self, given_dataset=None):
        """Transform the old format to the new format using HF datasets mapping. This is a one-time transformation for datasets which are in the old format.

        Args:
            given_dataset (Dataset, optional): The dataset to transform. Defaults to None. This is helpful for some older datasets which are loaded with custom code, but need to be transformed still.

        """
        if self.metadata.name not in OLD_FORMAT_RERANKING_TASKS:
            return

        logging.info(
            f"Transforming old format to standard format for {self.metadata.name}"
        )

        self.corpus = defaultdict(lambda: defaultdict(dict))
        self.queries = defaultdict(lambda: defaultdict(dict))
        self.relevant_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.top_ranked = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        hf_subsets = self.hf_subsets

        for hf_subset in hf_subsets:
            if given_dataset:
                cur_dataset = given_dataset
            elif "name" in self.metadata.dataset:
                cur_dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
                assert hf_subset == "default", (
                    f"Only default subset is supported for {self.metadata.name} since `name` is given in the metadata."
                )
            else:
                cur_dataset = datasets.load_dataset(
                    **self.metadata.dataset, name=hf_subset
                )  # type: ignore

            for split in cur_dataset:
                # Create an enumerated dataset to pass indices
                enumerated_dataset = Dataset.from_dict(
                    {
                        "index": range(len(cur_dataset[split])),
                        "query": cur_dataset[split]["query"],
                        "positive": cur_dataset[split]["positive"],
                        "negative": cur_dataset[split]["negative"],
                    }
                )

                # first, filter out the ones that have no positive or no negatives
                enumerated_dataset = enumerated_dataset.filter(
                    lambda example: len(example["positive"]) > 0
                    and len(example["negative"]) > 0
                )

                logger.info(
                    f"Filtered out {len(cur_dataset[split]) - len(enumerated_dataset)} examples with no positive or no negative examples. {len(enumerated_dataset)} examples remaining."
                )

                # Map the transformation function over the dataset
                processed_dataset = enumerated_dataset.map(
                    lambda example, idx: self.process_example(example, split, idx),
                    with_indices=True,
                    remove_columns=enumerated_dataset.column_names,
                )

                # Populate the data structures
                for item in processed_dataset:
                    query_id = item["query_id"]
                    self.queries[hf_subset][split][query_id] = item["query"]

                    # Add documents and relevance information
                    for doc_id, doc_text, relevance in zip(
                        item["doc_ids"], item["doc_texts"], item["relevance_scores"]
                    ):
                        self.corpus[hf_subset][split][doc_id] = {
                            "text": doc_text,
                            "_id": doc_id,
                        }
                        self.top_ranked[hf_subset][split][query_id].append(doc_id)
                        self.relevant_docs[hf_subset][split][query_id][doc_id] = (
                            relevance
                        )

        self.instructions = None
        self.data_loaded = True
