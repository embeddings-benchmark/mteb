from __future__ import annotations

import logging
from collections import defaultdict

import datasets
import tqdm

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
    """Abstract class for re-ranking experiments. This is mostly the same as the RetrievalEvaluator, but as previously it wasn't we need to keep it to transform old dataset versions into the same format.

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

    Old Format:
    -----------
    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        query: str
        positive: list[str]
        negative: list[str]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if self.metadata.name in OLD_FORMAT_RERANKING_TASKS:
            self.original_dataset = datasets.load_dataset(
                **self.metadata_dict["dataset"]
            )  # type: ignore
            self.transform_old_format_to_standard()
        else:
            # use AbsTaskRetrieval default to load the data
            # TODO: need to make sure top_ranked comes back
            return super().load_data(**kwargs)

    def transform_old_format_to_standard(self):
        """Transform the old format to the new format (see class doc string for details). Dataset has three features: query, positive, negative."""
        logging.info(
            f"Transforming old format to standard format for {self.metadata.name}"
        )
        self.corpus = defaultdict(dict)
        self.queries = defaultdict(dict)
        self.relevant_docs = defaultdict(lambda: defaultdict(dict))
        self.top_ranked = defaultdict(lambda: defaultdict(list))

        for split in self.original_dataset:
            # keep the lookups to prevent duplicate queries and documents for memory purposes
            corpus_lookup = {}
            query_lookup = {}
            for query_i in tqdm.tqdm(range(len(self.original_dataset[split]))):
                query: str = self.original_dataset[split]["query"][query_i]
                positive_docs: list[str] = self.original_dataset[split]["positive"][
                    query_i
                ]
                negative_docs: list[str] = self.original_dataset[split]["negative"][
                    query_i
                ]

                if query in query_lookup:
                    query_id = query_lookup[query]
                else:
                    query_id = f"{split}_query{query_i}"
                    query_lookup[query] = query_id
                self.queries[split][query_id] = query

                for i, pos_doc in enumerate(sorted(positive_docs)):
                    if pos_doc in corpus_lookup:
                        doc_id = corpus_lookup[pos_doc]
                    else:
                        doc_id = f"{query_id}_positive_{i}"
                        self.corpus[split][doc_id] = {"text": pos_doc, "_id": doc_id}
                        corpus_lookup[pos_doc] = doc_id

                    self.top_ranked[split][query_id].append(doc_id)
                    self.relevant_docs[split][query_id][doc_id] = 1

                for i, neg_doc in enumerate(sorted(negative_docs)):
                    if neg_doc in corpus_lookup:
                        doc_id = corpus_lookup[neg_doc]
                    else:
                        doc_id = f"{query_id}_negative_{i}"
                        self.corpus[split][doc_id] = {"text": neg_doc, "_id": doc_id}
                        corpus_lookup[neg_doc] = doc_id

                    self.top_ranked[split][query_id].append(doc_id)
                    self.relevant_docs[split][query_id][doc_id] = 0

        self.instructions = None  # previous tasks don't have instructions
        self.data_loaded = True

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        all_results = defaultdict(dict)
        max_docs = 0
        top_ranked = kwargs["top_ranked"]  # must be present for reranking
        for query_id in tqdm.tqdm(
            list(queries.keys()), leave=False, desc="Reranking over query-ids.."
        ):
            cur_queries = {query_id: queries[query_id]}
            if "instructions" in kwargs:
                instructions = kwargs["instructions"]
                cur_instructions = {queries[query_id]: instructions[queries[query_id]]}
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
