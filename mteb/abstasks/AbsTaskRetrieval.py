from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any, Callable

from datasets import Dataset, DatasetDict, concatenate_datasets

from mteb.models.models_protocols import Encoder
from mteb.types import (
    HFSubset,
    RelevantDocumentsType,
    ScoresDict,
)
from mteb.types.statistics import (
    DescriptiveStatistics,
    RelevantDocsStatistics,
    TextStatistics,
    TopRankedStatistics,
)

from ..create_dataloaders import (
    convert_conv_history_to_query,
    corpus_to_dict,
)
from ..evaluation.evaluators import RetrievalEvaluator
from ..evaluation.evaluators.retrieval_metrics import make_score_dict
from ._statistics_calculation import (
    calculate_relevant_docs_statistics,
    calculate_text_statistics,
    calculate_top_ranked_statistics,
)
from .AbsTask import AbsTask
from .retrieval_dataset_loaders import RetrievalDatasetLoader, RetrievalSplitData

logger = logging.getLogger(__name__)


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_samples: Number of queries and documents
        number_of_characters: Total number of characters in queries and documents

        documents_statistics: Statistics for documents
        queries_statistics: Statistics for queries
        relevant_docs_statistics: Statistics for relevant documents
        instructions_statistics: Statistics for instructions (if available)
        top_ranked_statistics: Statistics for top ranked documents (if available)
    """

    num_samples: int
    number_of_characters: int

    documents_statistics: TextStatistics
    queries_statistics: TextStatistics

    relevant_docs_statistics: RelevantDocsStatistics

    # these are for datasets with instructions
    instructions_statistics: TextStatistics | None

    # this is for datasets that do reranking
    top_ranked_statistics: TopRankedStatistics | None


class AbsTaskRetrieval(AbsTask):
    """Abstract class for retrieval experiments.

    Child-classes must implement the following properties:

    self.dataset: dict[str, dict[str, dict[str, Any]]]
        A dictionary containing all dataset components with the following structure:
        {
            hf_subset: {
                split: {
                    'corpus': dict[str, dict[str, str]],  # doc_id -> doc
                        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
                        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}
                    'queries': dict[str, str | list[str]]],  # query_id -> query
                        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, list[str]]] for conversations
                        E.g. {"test": {"q1": "query"}}
                        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}
                    'relevant_docs': dict[str, dict[str, int]],  # query_id -> doc_id -> score
                        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
                        E.g.: {"test": {"q1": {"document_one": 1}}}
                    'instructions': Optional[dict[str, str]],  # query_id -> instruction
                        Semantically, it should contain dict[split_name, dict[sample_id, list[doc_id]]] or dict[split_name, dict[sample_id, dict[doc_id, score]]]
                        E.g.: {"test": {"q1": ["document_one", "document_two"]}} or {"test": {"q1": {"document_one": 1, "document_two": 0.5}}}
                    'top_ranked': Optional[dict[str, list[str]]]  # query_id -> doc_ids
                        Semantically, it should contain dict[split_name, dict[sample_id, str]]. If there are multiple instructions per query, please duplicate the queries and give them unique ids for consolidation.
                        E.g. {"test": {"query-id1": "instruction text"}}
                }
            }
        }
    """

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."
    k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000]
    top_k: int = max(k_values)
    dataset: dict[str, dict[str, RetrievalSplitData]]
    cross_encoder_top_k: int = 100
    support_cross_encoder: bool = True
    support_search: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        empty_dataset = Dataset.from_dict({})
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus=empty_dataset,
                    queries=empty_dataset,
                    relevant_docs={},
                    instructions=None,
                    top_ranked=None,
                )
            )
        )

    def convert_v1_dataset_format_to_v2(self):
        # check if dataset is `v1` version
        if not hasattr(self, "queries"):
            return
        empty_dataset = Dataset.from_dict({})

        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus=empty_dataset,
                    queries=empty_dataset,
                    relevant_docs={},
                    instructions=None,
                    top_ranked=None,
                )
            )
        )

        if self.metadata.is_multilingual:
            for subset in self.queries:
                for split in self.queries[subset]:
                    queries = self.queries[subset][split]
                    corpus = self.corpus[subset][split]
                    self.dataset[subset][split]["queries"] = Dataset.from_list(
                        [{"id": k, "text": v} for k, v in queries.items()]
                    )
                    self.dataset[subset][split]["corpus"] = Dataset.from_list(
                        [
                            {
                                "id": k,
                                "text": v["text"],
                                "title": v.get("title", ""),
                            }
                            for k, v in corpus.items()
                        ]
                    )
                    self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                        subset
                    ][split]
                    if hasattr(self, "instructions"):
                        instructions = self.instructions[subset][split]
                        self.dataset[subset][split]["instructions"] = Dataset.from_list(
                            [{"id": k, "text": v} for k, v in instructions.items()]
                        )
                    if hasattr(self, "top_ranked"):
                        self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                            subset
                        ][split]
        else:
            subset = "default"
            for split in self.queries:
                queries = self.queries[split]
                corpus = self.corpus[split]
                self.dataset[subset][split]["queries"] = Dataset.from_list(
                    [{"id": k, "text": v} for k, v in queries.items()]
                )
                self.dataset[subset][split]["corpus"] = Dataset.from_list(
                    [
                        {
                            "id": k,
                            "text": v["text"],
                            "title": v.get("title", ""),
                        }
                        for k, v in corpus.items()
                    ]
                )
                self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                    split
                ].copy()
                if hasattr(self, "instructions"):
                    instructions = self.instructions[split]
                    self.dataset[subset][split]["instructions"] = Dataset.from_list(
                        [{"id": k, "text": v} for k, v in instructions.items()]
                    )
                if hasattr(self, "top_ranked"):
                    self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                        split
                    ].copy()

        del self.queries
        del self.corpus
        del self.relevant_docs
        if hasattr(self, "instructions"):
            del self.instructions
        if hasattr(self, "top_ranked"):
            del self.top_ranked

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]
        eval_splits = kwargs.get("eval_splits", self.metadata.eval_splits)
        trust_remote_code = self.metadata.dataset.get("trust_remote_code", False)
        revision = self.metadata.dataset["revision"]

        def process_data(split: str, hf_subset: str = "default"):
            """Helper function to load and process data for a given split and language"""
            logger.info(
                f"Loading {split} split for {hf_subset} subset of {self.metadata.name}"
            )

            self.dataset[hf_subset][split] = RetrievalDatasetLoader(
                hf_repo=dataset_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                split=split,
                config=hf_subset,
            ).load()

        if self.metadata.is_multilingual:
            for lang in self.metadata.eval_langs:
                for split in eval_splits:
                    process_data(split, lang)
        else:
            for split in eval_splits:
                process_data(split)
        self.dataset_transform()
        self.data_loaded = True

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate the model on the retrieval task.

        Args:
            model: Model to evaluate
            split: Split to evaluate on
            subsets_to_run: Optional list of subsets to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            **kwargs: Additional keyword arguments passed to the evaluator

        Returns:
            Dictionary mapping subsets to their evaluation scores
        """
        if not self.data_loaded:
            self.load_data()
        # TODO: convert all tasks directly https://github.com/embeddings-benchmark/mteb/issues/2030
        self.convert_v1_dataset_format_to_v2()

        return super().evaluate(
            model,
            split,
            subsets_to_run,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: RetrievalSplitData,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        # retrieval specific args
        save_predictions: bool = False,
        export_errors: bool = False,
        save_qrels: bool = False,
        output_folder: str = "results",
        results: dict[str, dict[str, float]] | None = None,
        **kwargs,
    ) -> ScoresDict:
        """Evaluate a model on a specific subset of the data.

        Args:
            model: Model to evaluate
            data_split: Data split to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            hf_split: Split to evaluate on
            hf_subset: Subset to evaluate on
            save_predictions: Whether to save predictions
            export_errors: Whether to export errors
            save_qrels: Whether to save the qrels
            output_folder: Folder to save the results
            results: Results from retrieval from previous run
            **kwargs: Additional keyword arguments passed to the evaluator

        Returns:
            Dictionary of evaluation scores
        """
        retriever = RetrievalEvaluator(
            corpus=data_split["corpus"],
            queries=data_split["queries"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            instructions=data_split["instructions"],
            top_ranked=data_split["top_ranked"],
            top_k=self.top_k,
            **kwargs,
        )

        if not results:
            start_time = time()
            results = retriever(
                model,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            end_time = time()
            logger.debug(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        if save_predictions or export_errors or save_qrels:
            output_folder = Path(output_folder)
            if not output_folder.exists():
                output_folder.mkdir(parents=True)

        if save_predictions:
            if self.top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[: self.top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        if save_qrels:
            with open(
                output_folder / f"{self.metadata.name}_{hf_subset}_qrels.json", "w"
            ) as f:
                json.dump(data_split["relevant_docs"], f)

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
        ) = retriever.evaluate(
            data_split["relevant_docs"],
            results,
            self.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        task_specific_scores = self.task_specific_scores(
            all_scores,
            data_split["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        scores = make_score_dict(
            ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, task_specific_scores
        )

        if export_errors:
            errors = {}

            if not save_predictions:
                for qid in results.keys():
                    doc_scores = results[qid]
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )[:1]
                    results[qid] = dict(sorted_docs)
            for qid, retrieved_docs in results.items():
                expected_docs = data_split["relevant_docs"][qid]
                false_positives = [
                    doc for doc in retrieved_docs if doc not in expected_docs
                ]
                false_negatives = [
                    doc for doc in expected_docs if doc not in retrieved_docs
                ]
                if false_positives or false_negatives:
                    errors[qid] = {
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }
            errors_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_errors.json"
            )
            with errors_save_path.open("w") as f:
                json.dump(errors, f)

        return scores

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return {}

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RetrievalDescriptiveStatistics:
        self.convert_v1_dataset_format_to_v2()
        if hf_subset and hf_subset in self.dataset:
            split_data = self.dataset[hf_subset][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            instructions = split_data["instructions"]
            top_ranked = split_data["top_ranked"]
        elif compute_overall:
            queries = None
            corpus = None
            instructions = None
            relevant_docs = {}
            top_ranked = {}
            for hf_subset in self.metadata.eval_langs:
                split_data = self.dataset[hf_subset][split]
                if queries is None:
                    queries = split_data["queries"]
                else:
                    queries = concatenate_datasets([queries, split_data["queries"]])
                if corpus is None:
                    corpus = split_data["corpus"]
                else:
                    corpus = concatenate_datasets([corpus, split_data["corpus"]])

                relevant_docs.update(
                    process_relevant_docs(split_data["relevant_docs"], hf_subset, split)
                )
                if (
                    "instructions" in split_data
                    and split_data["instructions"] is not None
                ):
                    if instructions is None:
                        instructions = split_data["instructions"]
                    else:
                        instructions = concatenate_datasets(
                            [instructions, split_data["instructions"]]
                        )

                if "top_ranked" in split_data and split_data["top_ranked"] is not None:
                    top_ranked.update(
                        {
                            f"{split}_{hf_subset}_{k}": v
                            for k, v in split_data["top_ranked"].items()
                        }
                    )
        else:
            if "default" in self.dataset and split != "default":
                return self._calculate_metrics_from_split(
                    split=split, hf_subset="default"
                )
            split_data = self.dataset["default"][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            instructions = split_data["instructions"]
            top_ranked = split_data["top_ranked"]

        corpus = corpus.map(corpus_to_dict)["text"]
        queries_texts = [q for q in queries["text"] if isinstance(q, str)]
        num_documents = len(corpus)
        num_queries = len(queries_texts)

        relevant_docs_statistics = calculate_relevant_docs_statistics(relevant_docs)

        if instructions is not None and len(instructions) > 0:
            instruction_statistics = calculate_text_statistics(
                instructions["instruction"]
            )
        else:
            instruction_statistics = None

        if top_ranked is not None and num_queries and len(top_ranked) > 0:
            top_ranked_statistics = calculate_top_ranked_statistics(
                top_ranked, num_queries
            )
        else:
            top_ranked_statistics = None

        corpus_statistics = calculate_text_statistics(corpus)
        if isinstance(queries["text"][0], (dict, list)):
            queries = queries.map(convert_conv_history_to_query)

        queries_statistics = calculate_text_statistics(queries["text"])

        number_of_characters = (
            corpus_statistics["total_text_length"]
            + queries_statistics["total_text_length"]
        )
        if instruction_statistics is not None:
            number_of_characters += instruction_statistics["total_text_length"]

        return RetrievalDescriptiveStatistics(
            num_samples=num_documents + num_queries,
            number_of_characters=number_of_characters,
            documents_statistics=corpus_statistics,
            queries_statistics=queries_statistics,
            relevant_docs_statistics=relevant_docs_statistics,
            instructions_statistics=instruction_statistics,
            top_ranked_statistics=top_ranked_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self.convert_v1_dataset_format_to_v2()

        def _push_section(
            data: dict[str, dict[Any, Any]],
            subset_item: str,
            hf_subset_name: str,
            converter: Callable[[Any, Any], dict[str, Any]] | None = None,
        ) -> None:
            """Helper function to push dataset

            Args:
                data: Dataset with all items
                subset_item: Select which part to take. E. g. corpus, queries etc
                hf_subset_name: Name of the current item on HF
                converter: Function to convert dict to datasets format
            """
            sections = {}
            for split in data.keys():
                # skip empty instructions and top ranked
                if subset_item not in data[split] or data[split][subset_item] is None:
                    continue
                if isinstance(sections[split], Dataset):
                    sections[split] = data[split][subset_item]
                elif converter is not None:
                    sections[split] = Dataset.from_list(
                        [
                            converter(idx, item)
                            for idx, item in data[split][subset_item].items()
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unexpected subset item type {subset_item} without converter"
                    )
            if len(sections) > 0:
                DatasetDict(sections).push_to_hub(repo_name, hf_subset_name)

        for subset in self.dataset:
            logger.info(f"Converting {subset} of {self.metadata.name}")
            _push_section(
                self.dataset[subset],
                "queries",
                f"{subset}-queries" if subset != "default" else "queries",
            )
            _push_section(
                self.dataset[subset],
                "corpus",
                f"{subset}-corpus" if subset != "default" else "corpus",
            )
            # Handle relevant_docs separately since one entry expands to multiple records.
            relevant_sections = {}
            for split, values in self.dataset[subset].items():
                relevant_docs = values["relevant_docs"]
                entries = []
                for query_id, docs in relevant_docs.items():
                    for doc_id, score in docs.items():
                        entries.append(
                            {
                                "query-id": query_id,
                                "corpus-id": doc_id,
                                "score": score,
                            }
                        )
                relevant_sections[split] = Dataset.from_list(entries)
            DatasetDict(relevant_sections).push_to_hub(
                repo_name, f"{subset}-qrels" if subset != "default" else "qrels"
            )

            _push_section(
                self.dataset[subset],
                "instructions",
                f"{subset}-instruction" if subset != "default" else "instruction",
            )
            _push_section(
                self.dataset[subset],
                "top_ranked",
                f"{subset}-top_ranked" if subset != "default" else "top_ranked",
                lambda idx, docs: {"query-id": idx, "corpus-ids": docs},
            )


def process_relevant_docs(
    collection: dict[str, dict[str, float]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, float]]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return_collection = {}
    for query_id, relevant in collection.items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = {
            f"{split}_{hf_subset}_{doc_id}": value for doc_id, value in relevant.items()
        }
    return return_collection
