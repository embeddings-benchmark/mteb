from __future__ import annotations

import logging
from collections import defaultdict

import datasets
import tqdm
from datasets import Dataset

from ..load_results.task_results import ScoresDict
from .AbsTaskRetrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)

OLD_FORMAT_RERANKING_TASKS = [
    "AskUbuntuDupQuestions",
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
]


class AbsTaskReranking(AbsTaskRetrieval):
    """Abstract class for re-ranking experiments. This is mostly the same as the RetrievalEvaluator, but treats each query as a "mini" retrieval problem.

    New Format:
    -----------
    Same as AbsTaskRetrieval, but with a top_ranked file that contains the passages to rerank. The dataset should contain the following columns:

        self.corpus: dict[str, dict[str, str]]
            Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
            E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

        self.queries: dict[str, dict[str, Union[str, list[str]]]]
            Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, list[str]]] for conversations
            E.g. {"test": {"q1": "query"}}
            or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

        self.relevant_docs: dict[str, dict[str, dict[str, int]]]
            Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
            E.g.: {"test": {"q1": {"document_one": 1}}}

        self.top_ranked: dict[str, dict[str, list[str]]] or dict[str, dict[str, dict[str, float]]]
            Semantically, it should contain dict[split_name, dict[sample_id, list[doc_id]]] or dict[split_name, dict[sample_id, dict[doc_id, score]]]
            E.g.: {"test": {"q1": ["document_one", "document_two"]}} or {"test": {"q1": {"document_one": 1, "document_two": 0.5}}}
    """

    def __init__(self, **kwargs):
        super(AbsTaskRetrieval, self).__init__(**kwargs)

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
            # have "a" in front so that positives are first, then negatives
            #   this shouldn't matter except for ties, and the previous reranking results
            #   had the positives first
            doc_id = f"apositive_{i}_{query_id}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(pos_doc)
            example_data["relevance_scores"].append(1)

        for i, neg_doc in enumerate(negative_docs):
            doc_id = f"negative_{i}_{query_id}"
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

        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            if given_dataset:
                cur_dataset = given_dataset
            elif "name" in self.metadata_dict["dataset"]:
                cur_dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore
                assert (
                    hf_subset == "default"
                ), f"Only default subset is supported for {self.metadata.name} since `name` is given in the metadata."
            else:
                cur_dataset = datasets.load_dataset(
                    **self.metadata_dict["dataset"], name=hf_subset
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
                    lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0
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

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        """Evaluate each query_id as a "mini" retrieval corpus, and rerank the top-ranked documents for each query_id."""
        all_results = defaultdict(dict)
        max_docs = 0
        top_ranked = kwargs["top_ranked"]  # must be present for reranking
        for query_id in tqdm.tqdm(
            list(queries.keys()), leave=False, desc="Reranking over query-ids.."
        ):
            cur_queries = {query_id: queries[query_id]}
            if "instructions" in kwargs:
                instructions = kwargs["instructions"]
                cur_instructions = {query_id: instructions[query_id]}
            else:
                cur_instructions = None

            doc_ids_to_rerank = top_ranked[query_id]
            cur_corpus = {doc_id: corpus[doc_id] for doc_id in doc_ids_to_rerank}
            if (
                len(cur_corpus) > max_docs
            ):  # use this to make sure we get the correct MAP/MRR at max length
                max_docs = len(cur_corpus)

            # to handle instruction-based reranking we pass both query_id and instructions (unused if not instruction-based)
            results = retriever(
                cur_corpus,
                cur_queries,
                instructions=cur_instructions,
                query_id=query_id,
            )
            # results should have only one key, the query_id
            all_results[query_id] = results[query_id]

        # do the evaluation like normal now, but pass our results
        if max_docs > max(retriever.k_values):
            # only added if we need a large k-value for reranking past 1000
            retriever.k_values += [max_docs]

        return super()._evaluate_subset(
            retriever,
            corpus,
            queries,
            relevant_docs,
            hf_subset,
            results=all_results,
            **kwargs,
        )
