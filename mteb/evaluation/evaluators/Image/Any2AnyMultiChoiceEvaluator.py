from __future__ import annotations

import heapq
import io
import json
import logging
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pytrec_eval
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder
from mteb.requires_package import requires_image_dependencies

from ..Evaluator import Evaluator
from ..utils import (
    confidence_scores,
    cos_sim,
    dot_score,
    download,
    hole,
    mrr,
    nAUC,
    recall_cap,
    top_k_accuracy,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def get_default_transform():
    requires_image_dependencies()
    from torchvision import transforms

    return transforms.Compose([transforms.PILToTensor()])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            # Assume the image is already in a usable format (e.g., PIL Image)
            image = image
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def custom_collate_fn(batch):
    return batch


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class Any2AnyMultiChoiceSearch:
    def __init__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any] = {},
        corpus_chunk_size: int = 20000,
        previous_results: str | None = None,
        **kwargs: Any,
    ):
        # Model is class that provides get_text_embeddings() and get_image_embeddings()
        self.model = model
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 128

        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.previous_results = previous_results
        self.batch_size = encode_kwargs.get("batch_size")
        self.show_progress_bar = encode_kwargs.get("show_progress_bar")
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = defaultdict(list)
        self.results = {}

        if self.previous_results is not None:
            self.previous_results = self.load_results_file()

    def search(
        self,
        corpus: Dataset,  # solve memoery issues
        queries: Dataset,  # solve memoery issues
        qrels: Dataset,
        top_k: int,
        score_function: str,
        task_name: str | None = None,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
            )

        logger.info("Encoding Queries.")
        query_ids = list(queries["id"])
        self.results = {qid: {} for qid in query_ids}

        q_modality = queries[0]["modality"]

        default_transform = get_default_transform()

        if q_modality == "text":
            query_texts = queries["text"]
            query_embeddings = self.model.get_text_embeddings(
                texts=query_texts,
                task_name=task_name,
                batch_size=self.encode_kwargs["batch_size"],
            )
        else:
            queries_dataset = ImageDataset(
                queries, image_column_name="image", transform=default_transform
            )
            query_image_dataloader = DataLoader(
                queries_dataset,
                batch_size=self.encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 16),
            )
            if q_modality == "image":
                query_embeddings = self.model.get_image_embeddings(
                    images=query_image_dataloader,
                    batch_size=self.encode_kwargs["batch_size"],
                    task_name=task_name,
                )
            elif q_modality == "image,text":
                query_texts = queries["text"]
                query_embeddings = self.model.get_fused_embeddings(
                    texts=query_texts,
                    images=query_image_dataloader,
                    batch_size=self.encode_kwargs["batch_size"],
                    task_name=task_name,
                )
            else:
                raise ValueError(f"Unsupported modality: {q_modality}")

        logger.info("Preparing Corpus...")
        corpus_ids = list(corpus["id"])

        corpus_modality = corpus[0]["modality"]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(
            f"Scoring Function: {self.score_function_desc[score_function]} ({score_function})"
        )

        result_heaps = {qid: [] for qid in query_ids}
        for chunk_start in range(0, len(corpus), self.corpus_chunk_size):
            chunk = corpus.select(
                range(
                    chunk_start, min(chunk_start + self.corpus_chunk_size, len(corpus))
                )
            )
            chunk_ids = corpus_ids[chunk_start : chunk_start + self.corpus_chunk_size]

            if corpus_modality == "text":
                corpus_texts = chunk["text"]
                sub_corpus_embeddings = self.model.get_text_embeddings(
                    texts=corpus_texts, batch_size=self.encode_kwargs["batch_size"]
                )
            else:
                corpus_dataset = ImageDataset(
                    chunk, image_column_name="image", transform=default_transform
                )
                corpus_image_dataloader = DataLoader(
                    corpus_dataset,
                    batch_size=self.encode_kwargs["batch_size"],
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=min(math.floor(os.cpu_count() / 2), 16),
                )
                if corpus_modality == "image":
                    sub_corpus_embeddings = self.model.get_image_embeddings(
                        images=corpus_image_dataloader,
                        batch_size=self.encode_kwargs["batch_size"],
                        task_name=task_name,
                    )
                elif corpus_modality == "image,text":
                    corpus_texts = chunk["text"]
                    sub_corpus_embeddings = self.model.get_fused_embeddings(
                        texts=corpus_texts,
                        images=corpus_image_dataloader,
                        batch_size=self.encode_kwargs["batch_size"],
                        task_name=task_name,
                    )
                else:
                    raise ValueError(f"Unsupported modality: {corpus_modality}")

            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            for query_idx in range(len(query_embeddings)):
                query_id = query_ids[query_idx]
                # discount answers which aren't a multiple choice (where there is a qrel entry for both query and corpus id)
                for c_idx, c_id in enumerate(chunk_ids):
                    if c_id not in qrels[query_id]:
                        cos_scores[query_idx, c_idx] = -1

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, cos_scores.size(1)),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = chunk_ids[sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        if "https://" in self.previous_results:
            # download the file
            if not os.path.exists(self.previous_results):
                url_descriptor = self.previous_results.split("https://")[-1].replace(
                    "/", "--"
                )
                dest_file = os.path.join(
                    "results", f"cached_predictions--{url_descriptor}"
                )
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                download(self.previous_results, dest_file)
                logger.info(
                    f"Downloaded the previous results at {self.previous_results} to {dest_file}"
                )
            self.previous_results = dest_file

        with open(self.previous_results) as f:
            previous_results = json.load(f)
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results


class Any2AnyMultiChoiceEvaluator(Evaluator):
    def __init__(
        self,
        retriever=None,
        task_name: str | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
        score_function: str = "cos_sim",
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retriever = Any2AnyMultiChoiceSearch(
            retriever, encode_kwargs=encode_kwargs, **kwargs
        )
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.score_function = score_function
        self.task_name = task_name

    def __call__(
        self,
        corpus: dict[str, dict[str, str | Image.Image]],
        queries: dict[str, dict[str, str | Image.Image]],
        qrels: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        return self.retriever.search(
            corpus,
            queries,
            qrels,
            self.top_k,
            self.score_function,
            task_name=self.task_name,  # type: ignore
        )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
        skip_first_result: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        if ignore_identical_ids:
            logger.debug(
                "For evaluation, ``ignore_identical_ids=True`` is set to True, the evaluator will ignore identical query and document ids."
            )
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=True`` to ignore this."
            )

        all_ndcgs, all_aps, all_recalls, all_precisions, all_cv_recalls = (
            {},
            {},
            {},
            {},
            {},
        )

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []
            all_cv_recalls[f"CV_Recall@{k}"] = []  # (new) CV-style Recall

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        sorted_results = {
            qid: sorted(rels.items(), key=lambda item: item[1], reverse=True)
            for qid, rels in results.items()
        }

        if skip_first_result:
            for qid, rels in sorted_results.items():
                sorted_results[qid].pop(0)

        for query_id in scores.keys():
            top_docs = [
                doc_id for doc_id, _ in sorted_results.get(query_id, [])
            ]  # Sorted list of doc IDs
            # we need to discount qrels that have a ground truth score of 0
            relevant_docs = {
                key
                for key in qrels.get(query_id, {}).keys()
                if qrels[query_id][key] != 0
            }

            for k in k_values:
                top_k_docs = top_docs[:k]
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

                if relevant_docs.intersection(top_k_docs):
                    all_cv_recalls[f"CV_Recall@{k}"].append(1.0)
                else:
                    all_cv_recalls[f"CV_Recall@{k}"].append(0.0)

        ndcg, _map, recall, precision, cv_recall = (
            all_ndcgs.copy(),
            all_aps.copy(),
            all_recalls.copy(),
            all_precisions.copy(),
            all_cv_recalls.copy(),
        )

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
            precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)
            cv_recall[f"CV_Recall@{k}"] = round(
                sum(cv_recall[f"CV_Recall@{k}"]) / len(scores), 5
            )

        naucs = Any2AnyMultiChoiceEvaluator.evaluate_abstention(
            results,
            {**all_ndcgs, **all_aps, **all_recalls, **all_precisions, **all_cv_recalls},
        )

        return ndcg, _map, recall, precision, cv_recall, naucs

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
        output_type: str = "all",
    ) -> tuple[dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            metric_scores = recall_cap(qrels, results, k_values, output_type)

        elif metric.lower() in ["hole", "hole@k"]:
            metric_scores = hole(qrels, results, k_values, output_type)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            metric_scores = top_k_accuracy(qrels, results, k_values, output_type)

        naucs = Any2AnyMultiChoiceEvaluator.evaluate_abstention(results, metric_scores)
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}

        return metric_scores_avg, naucs

    @staticmethod
    def evaluate_abstention(
        results: dict[str, dict[str, float]],
        metric_scores: dict[str, list[float]],
    ) -> dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997"""
        all_sim_scores = [list(results[qid].values()) for qid in list(results.keys())]
        all_conf_scores = [
            confidence_scores(sim_scores) for sim_scores in all_sim_scores
        ]
        conf_fcts = list(all_conf_scores[0].keys())
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        metric_scores = {k: np.array(v) for k, v in metric_scores.items()}
        naucs = {}

        for metric_name, scores in metric_scores.items():
            for fct, conf_scores in all_conf_scores.items():
                naucs[f"nAUC_{metric_name}_{fct}"] = nAUC(conf_scores, scores)

        return naucs

    @staticmethod
    def calculate_cv_style_recall(
        qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k: int
    ) -> dict[str, float]:
        """Calculate CV-style recall: Recall is 1 if any relevant document is
        retrieved in the top k, otherwise 0.
        """
        cv_recalls = {}
        for query_id, relevant_docs in qrels.items():
            retrieved_docs = list(results.get(query_id, {}).keys())[
                :k
            ]  # Retrieve top k documents
            if any(doc_id in relevant_docs for doc_id in retrieved_docs):
                cv_recalls[query_id] = (
                    1.0  # If any relevant doc is found in top k, recall is 1
                )
            else:
                cv_recalls[query_id] = 0.0  # Otherwise, recall is 0
        return cv_recalls
