from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from sklearn.metrics import auc


def cos_sim(a, b):
    """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """  # noqa: D402
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = []

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            rr = 0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            MRR[f"MRR@{k}"].append(rr)

    if output_type == "mean":
        for k in k_values:
            MRR[f"MRR@{k}"] = round(sum(MRR[f"MRR@{k}"]) / len(qrels), 5)
            logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    elif output_type == "all":
        pass

    return MRR


def recall_cap(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = []

    k_max = max(k_values)
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        query_relevant_docs = [
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        ]
        for k in k_values:
            retrieved_docs = [
                row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0
            ]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"].append(len(retrieved_docs) / denominator)

    if output_type == "mean":
        for k in k_values:
            capped_recall[f"R_cap@{k}"] = round(
                sum(capped_recall[f"R_cap@{k}"]) / len(qrels), 5
            )
            logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    elif output_type == "all":
        pass

    return capped_recall


def hole(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = []

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logging.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            Hole[f"Hole@{k}"].append(len(hole_docs) / k)

    if output_type == "mean":
        for k in k_values:
            Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)
            logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    elif output_type == "all":
        pass

    return Hole


def top_k_accuracy(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = []

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"].append(1.0)
                    break

    if output_type == "mean":
        for k in k_values:
            top_k_acc[f"Accuracy@{k}"] = round(
                top_k_acc[f"Accuracy@{k}"] / len(qrels), 5
            )
            logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    elif output_type == "all":
        pass

    return top_k_acc


def get_rank_from_dict(
    dict_of_results: dict[str, float], doc_id: str
) -> Tuple[int, float]:
    tuple_of_id_score = dict_of_results.items()
    sorted_by_score = sorted(tuple_of_id_score, key=lambda x: x[1], reverse=True)
    for i, (id, score) in enumerate(sorted_by_score):
        if id == doc_id:
            return i + 1, score

    return len(sorted_by_score) + 1, 0


def evaluate_change(
    original_run: dict[str, dict[str, float]],
    new_run: dict[str, dict[str, float]],
    changed_qrels: dict[str, List[str]],
) -> dict[str, float]:
    changes = []
    for qid in changed_qrels.keys():
        original_qid_run = original_run[qid]
        new_qid_run = new_run[qid]
        for idx, changed_doc in enumerate(changed_qrels[qid]):
            original_rank, original_score = get_rank_from_dict(
                original_qid_run, changed_doc
            )
            new_rank, new_score = get_rank_from_dict(new_qid_run, changed_doc)
            change = int(original_rank - new_rank)
            changes.append(
                {
                    "qid": qid,
                    "doc_id": changed_doc,
                    "change": change,
                    "relevance": 0,
                    "og_rank": original_rank,
                    "new_rank": new_rank,
                    "og_score": original_score,
                    "new_score": new_score,
                }
            )

    # we now have a DF of [qid, doc_id, change] to run our calculations with
    changes_df = pd.DataFrame(changes)
    changes_df["p-MRR"] = changes_df.apply(lambda x: rank_score(x), axis=1)
    qid_wise = changes_df.groupby("qid").agg({"p-MRR": "mean"})
    return {
        "p-MRR": qid_wise["p-MRR"].mean(),
    }


def rank_score(x: dict[str, float]) -> float:
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    else:
        return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))


# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm.tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def confidence_scores(sim_scores: List[float]) -> Dict[str, float]:
    """Computes confidence scores for a single instance = (query, positives, negatives)

    Args:
        sim_scores: Query-documents similarity scores with length `num_pos+num_neg`

    Returns:
        conf_scores:
            - `max`: Maximum similarity score
            - `std`: Standard deviation of similarity scores
            - `diff1`: Difference between highest and second highest similarity scores
    """
    sim_scores_sorted = sorted(sim_scores)[::-1]

    cs_max = sim_scores_sorted[0]
    cs_std = np.std(sim_scores)
    if len(sim_scores) > 1:
        cs_diff1 = sim_scores_sorted[0] - sim_scores_sorted[1]
    elif len(sim_scores) == 1:
        cs_diff1 = 0.0

    conf_scores = {"max": cs_max, "std": cs_std, "diff1": cs_diff1}

    return conf_scores


def nAUC(
    conf_scores: np.ndarray,
    metrics: np.ndarray,
    abstention_rates: np.ndarray = np.linspace(0, 1, 11)[:-1],
) -> float:
    """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997
    1/ Computes the raw abstention curve, i.e., the average evaluation metric at different abstention rates determined by the confidence scores
    2/ Computes the oracle abstention curve, i.e., the best theoretical abstention curve (e.g.: at a 10% abstention rate, the oracle abstains on the bottom-10% instances with regard to the evaluation metric)
    3/ Computes the flat abstention curve, i.e., the one remains flat for all abstention rates (ineffective abstention)
    4/ Computes the area under the three curves
    5/ Finally scales the raw AUC between the oracle and the flat AUCs to get normalized AUC

    Args:
        conf_scores: Instance confidence scores used for abstention thresholding, with shape `(num_test_instances,)`
        metrics: Metric evaluations at instance-level (e.g.: average precision, NDCG...), with shape `(num_test_instances,)`
        abstention_rates: Target rates for the computation of the abstention curve

    Returns:
        abst_nauc: Normalized area under the abstention curve (upper-bounded by 1)
    """

    def abstention_curve(
        conf_scores: np.ndarray,
        metrics: np.ndarray,
        abstention_rates: np.ndarray = np.linspace(0, 1, 11)[:-1],
    ) -> np.ndarray:
        """Computes the raw abstention curve for a given set of evaluated instances and corresponding confidence scores

        Args:
            conf_scores: Instance confidence scores used for abstention thresholding, with shape `(num_test_instances,)`
            metrics: Metric evaluations at instance-level (e.g.: average precision, NDCG...), with shape `(num_test_instances,)`
            abstention_rates: Target rates for the computation of the abstention curve

        Returns:
            abst_curve: Abstention curve of length `len(abstention_rates)`
        """
        conf_scores_argsort = np.argsort(conf_scores)
        abst_curve = np.zeros(len(abstention_rates))

        for i, rate in enumerate(abstention_rates):
            num_instances_abst = min(
                round(rate * len(conf_scores_argsort)), len(conf_scores) - 1
            )
            abst_curve[i] = metrics[conf_scores_argsort[num_instances_abst:]].mean()

        return abst_curve

    abst_curve = abstention_curve(conf_scores, metrics, abstention_rates)
    or_curve = abstention_curve(metrics, metrics, abstention_rates)
    abst_auc = auc(abstention_rates, abst_curve)
    or_auc = auc(abstention_rates, or_curve)
    flat_auc = or_curve[0] * (abstention_rates[-1] - abstention_rates[0])

    if or_auc == flat_auc:
        abst_nauc = np.nan
    else:
        abst_nauc = (abst_auc - flat_auc) / (or_auc - flat_auc)

    return abst_nauc
