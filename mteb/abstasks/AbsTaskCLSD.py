from __future__ import annotations

import json
import logging
import os
from time import time

from ..evaluation.evaluators import RetrievalEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskCLSD(AbsTask):
    """Abstract class for Cross Lingual Semantic Discrimination Tasks. Initialised from starting code of AbsTaskRetrieval

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]]
        E.g. {"test": {"q1": "query"}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """Not implemented, create within the instance. This function must populate:
        self.corpus: Distractors (Negative) and Parallel (Positive) samples of a search.
        self.queries: The source text to search with.
        self.relevant_docs: Denotes which are the relevant documents (The positive).
        For an example implementation and layout of the huggingface dataset, please look at the CrossLingualSemanticDiscriminationWMT19.py
        """
        raise NotImplementedError

    def evaluate(self, model, split="test", **kwargs):
        retriever = RetrievalEvaluator(model, **kwargs)

        scores = {}
        if self.is_crosslingual or self.is_multilingual:
            for lang in self.langs:
                logger.info(f"Language: {lang}")
                corpus, queries, relevant_docs = (
                    self.corpus[lang][split],
                    self.queries[lang][split],
                    self.relevant_docs[lang][split],
                )
                scores[lang] = self._evaluate_split(
                    retriever, corpus, queries, relevant_docs, lang, **kwargs
                )
        else:
            corpus, queries, relevant_docs = (
                self.corpus[split],
                self.queries[split],
                self.relevant_docs[split],
            )
            scores = self._evaluate_split(
                retriever, corpus, queries, relevant_docs, None, **kwargs
            )
        return scores

    def _evaluate_split(  # Changed from AbsTaskRetrieval to only calculate exact accuracy.
        self, retriever, corpus, queries, relevant_docs, lang=None, **kwargs
    ):
        start_time = time()
        results = retriever(corpus, queries)
        end_time = time()
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )
        if kwargs.get("save_predictions", False):
            output_folder = kwargs.get("output_folder", "results")
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            top_k = kwargs.get("top_k", None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            if lang is None:
                qrels_save_path = (
                    f"{output_folder}/{self.metadata_dict['name']}_predictions.json"
                )
            else:
                qrels_save_path = f"{output_folder}/{self.metadata_dict['name']}_{lang}_predictions.json"

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)
        # Key difference of function is here, we only use evaluate custom from the retriever and make a smaller table.
        accuracy = retriever.evaluate_custom(
            qrels=relevant_docs, results=results, k_values=[1], metric="accuracy"
        )
        scores = {"accuracy": v for (k, v) in accuracy.items()}
        return scores

    def calculate_metadata_metrics(self) -> None:
        self.load_data()

        for split in self.metadata_dict["eval_splits"]:
            if self.is_multilingual:
                for lang in self.relevant_docs.keys():
                    process_language(
                        self.relevant_docs[lang][split],
                        self.queries[lang][split],
                        self.corpus[lang][split],
                        lang,
                    )
            else:
                process_language(
                    self.relevant_docs[split], self.queries[split], self.corpus[split]
                )


def process_language(relevant_docs, queries, corpus, lang=None):
    total_length, num_pairs = calculate_length_and_count(relevant_docs, queries, corpus)
    average_length = total_length / num_pairs if num_pairs else 0
    num_documents = len(queries) + len(corpus)

    language_description = f" for language {lang}" if lang else ""
    print(f"Average character length{language_description} is {average_length}")
    print(f"Number of queries and documents{language_description} is {num_documents}")


def calculate_length_and_count(relevant_docs, queries, corpus):
    total_length = 0
    num_pairs = 0
    for query_id, docs in relevant_docs.items():
        query = queries[query_id]
        for doc_id in docs:
            # not relevant
            if docs[doc_id] == 0:
                continue
            doc = corpus[doc_id]
            doc_text = doc["title"] + doc["text"]
            total_length += len(query) + len(doc_text)
            num_pairs += 1
    return total_length, num_pairs
