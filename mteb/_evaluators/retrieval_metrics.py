import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytrec_eval
from packaging.version import Version
from sklearn.metrics import auc

from mteb.types import RelevantDocumentsType, RetrievalEvaluationResult

logger = logging.getLogger(__name__)


def mrr(
    qrels: RelevantDocumentsType,
    results: Mapping[str, Mapping[str, float]],
    k_values: list[int],
) -> dict[str, list[float]]:
    mrr_metrics = defaultdict(list)

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
        for k in k_values:
            rr = 0.0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_metrics[f"MRR@{k}"].append(rr)
    return mrr_metrics


def recall_cap(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, list[float | None]]:
    capped_recall: dict[str, list[float | None]] = defaultdict(list)

    k_max = max(k_values)

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
            if denominator == 0:
                capped_recall[f"R_cap_at_{k}"].append(None)
            capped_recall[f"R_cap_at_{k}"].append(len(retrieved_docs) / denominator)
    return capped_recall


def hole(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, list[float]]:
    hole = defaultdict(list)

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            hole[f"Hole_at_{k}"].append(len(hole_docs) / k)
    return hole


def top_k_accuracy(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, list[float]]:
    top_k_acc = defaultdict(list)

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"].append(1.0)
                    break
    return top_k_acc


def get_rank_from_dict(
    dict_of_results: dict[str, float], doc_id: str
) -> tuple[int, float]:
    tuple_of_id_score = dict_of_results.items()
    sorted_by_score = sorted(tuple_of_id_score, key=lambda x: x[1], reverse=True)
    for i, (id, score) in enumerate(sorted_by_score):
        if id == doc_id:
            return i + 1, score

    return len(sorted_by_score) + 1, 0


def calculate_pmrr(original_run, new_run, changed_qrels):
    changes = []
    for qid in changed_qrels.keys():
        if qid + "-og" not in original_run or qid + "-changed" not in new_run:
            logging.warning(f"Query {qid} not found in the runs for calculating p-MRR")
            continue
        original_qid_run = original_run[qid + "-og"]
        new_qid_run = new_run[qid + "-changed"]
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
    return qid_wise["p-MRR"].mean()


def evaluate_p_mrr_change(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    changed_qrels: dict[str, list[str]],
    k_values: list[int],
) -> dict[str, float | dict[str, float]]:
    """Computes the scores needed for FollowIR datasets.

    Including p-MRR (measuring change in instruction) and details about the original instruction run and changed instruction run. Used by IntructionRetrieval/Reranking tasks.

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        changed_qrels: A mapping from query IDs (without -og or -changed) to a list of document IDs that have changed relevance
        k_values: The k values for which to compute the scores

    Returns:
        A dictionary with the scores, including "p-MRR", "og" and "changed" keys.
    """
    followir_scores: dict[str, float | dict[str, float]] = defaultdict(dict)

    qrels_sep = {
        "og": {k: v for k, v in qrels.items() if k.endswith("-og")},
        "changed": {k: v for k, v in qrels.items() if not k.endswith("-og")},
    }

    original_run = {}
    new_run = {}
    # make original run from the results file with all "-og" items only and vice versa
    for qid, docs in results.items():
        if qid.endswith("-og"):
            original_run[qid] = docs
        else:
            new_run[qid] = docs

    p_mrr = calculate_pmrr(original_run, new_run, changed_qrels)
    followir_scores["p-MRR"] = p_mrr

    # unfortunately, have to re-compute scores here to get only og and changed scores
    followir_scores["og"] = {}
    followir_scores["changed"] = {}
    for name, group in [("og", original_run), ("changed", new_run)]:
        (
            scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            avg_mrr,
            naucs_mrr,
            cv_recall,
        ) = calculate_retrieval_scores(group, qrels_sep[name], k_values)
        # add these to the followir_scores with name prefix
        scores_dict = make_score_dict(
            ndcg, _map, recall, precision, naucs, avg_mrr, naucs_mrr, cv_recall, {}
        )
        for key, value in scores_dict.items():
            followir_scores[name][key] = value  # type: ignore[index]

    return followir_scores


def rank_score(x: dict[str, float]) -> float:
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    else:
        return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))


def confidence_scores(sim_scores: list[float]) -> dict[str, float]:
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
    cs_std = float(np.std(sim_scores))
    cs_diff1 = 0.0
    if len(sim_scores) > 1:
        cs_diff1 = sim_scores_sorted[0] - sim_scores_sorted[1]
    elif len(sim_scores) == 1:
        cs_diff1 = 0.0

    conf_scores = {"max": cs_max, "std": cs_std, "diff1": cs_diff1}
    return conf_scores


def nauc(
    conf_scores: np.ndarray,
    metrics: np.ndarray,
    abstention_rates: np.ndarray = np.linspace(0, 1, 11)[:-1],
) -> float:
    """Computes normalized Area Under the Curve (nAUC) on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997

    1. Computes the raw abstention curve, i.e., the average evaluation metric at different abstention rates determined by the confidence scores
    2. Computes the oracle abstention curve, i.e., the best theoretical abstention curve (e.g.: at a 10% abstention rate, the oracle abstains on the bottom-10% instances with regard to the evaluation metric)
    3. Computes the flat abstention curve, i.e., the one remains flat for all abstention rates (ineffective abstention)
    4. Computes the area under the three curves
    5. Finally scales the raw AUC between the oracle and the flat AUCs to get normalized AUC

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
        # argsort stable=True is default in numpy >2.0.0
        if Version(np.__version__) < Version("2.0.0"):
            conf_scores_argsort = np.argsort(conf_scores)
        else:
            conf_scores_argsort = np.argsort(conf_scores, stable=True)
        abst_curve = np.zeros(len(abstention_rates))

        for i, rate in enumerate(abstention_rates):
            num_instances_abst = min(
                round(rate * conf_scores_argsort.shape[0]), len(conf_scores) - 1
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


def paired_accuracy(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    scores: dict[str, dict[str, float]],
) -> float:
    """Computes the paired accuracy.

    This means both queries for an instance have to be correct for it to count.
    This is because models will prefer one passage all the time, giving it 50% automatically unless we correct for this.
    For more details, see https://arxiv.org/abs/2305.07614

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        scores: The scores for the queries, to extract top_1 recall for each query

    Returns:
        The paired accuracy score
    """
    # group the queries by the query id
    query_keys = set()
    for key in qrels.keys():
        query_keys.add(key.split("_")[0])

    paired_scores = []
    for key in query_keys:
        # get recall_at_1 for both q1 and q2
        q1_recall_at_1 = scores[f"{key}_q1"]["recall_1"]
        q2_recall_at_1 = scores[f"{key}_q2"]["recall_1"]

        # the score is 1 if both are 1, 0 otherwise
        paired_scores.append(1 if q1_recall_at_1 == 1 and q2_recall_at_1 == 1 else 0)

    return sum(paired_scores) / len(paired_scores)


def robustness_at_10(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    scores: dict[str, dict[str, float]],
) -> float:
    """Computes the robustness at 10. This computes the lowest ndcg@10 over all instructions. Taken from https://arxiv.org/abs/2402.14334

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        scores: The scores for the queries, to extract ndcg@10 for each query

    Returns:
        The robustness at 10 score
    """
    query_keys = defaultdict(list)
    for key in qrels.keys():
        query_keys[key.split("_")[0]].append(key)

    robustness_scores = []
    for _, keys in query_keys.items():
        # get the ndcg@10 for each query
        current_scores = []
        for key in keys:
            current_scores.append(scores[key]["ndcg_cut_10"])

        # get the lowest ndcg@10
        robustness_scores.append(min(current_scores))

    return sum(robustness_scores) / len(robustness_scores)


def make_score_dict(
    ndcg: dict[str, float],
    _map: dict[str, float],
    recall: dict[str, float],
    precision: dict[str, float],
    mrr: dict[str, float],
    naucs: dict[str, float],
    naucs_mrr: dict[str, float],
    cv_recall: dict[str, float],
    task_scores: dict[str, float],
    previous_results_model_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        # For MTEB multichoice tasks, we report recall@1 as the main metric.
        # This follows how MTEB implements these tasks, and recall@1 here is equivalent to accuracy.
        "accuracy": recall["Recall@1"],
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        **{
            k.replace("@", "_at_").replace("_P", "_precision").lower(): v
            for k, v in naucs.items()
        },
        **{
            k.replace("@", "_at_").replace("_P", "_precision").lower(): v
            for k, v in naucs_mrr.items()
        },
        **{f"cv_recall_at_{k.split('@')[1]}": v for k, v in cv_recall.items()},
        **task_scores,
        **(
            {"previous_results_model_meta": previous_results_model_meta}
            if previous_results_model_meta
            else {}
        ),
    }


def parse_metrics_from_scores(
    scores: dict[str, dict[str, float]], k_values: list[int]
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    all_ndcgs, all_aps, all_recalls, all_precisions = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = {}, {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(all_ndcgs[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(all_aps[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(all_recalls[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(all_precisions[f"P@{k}"]) / len(scores), 5)

    return (
        ndcg,
        _map,
        recall,
        precision,
        all_ndcgs,
        all_aps,
        all_recalls,
        all_precisions,
    )


def max_over_subqueries(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    """Computes the max over subqueries scores when merging.

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        k_values: The k values for which to compute the scores

    Returns:
        A dictionary with the scores, prefixed with "max_over_subqueries_"
    """
    query_keys = defaultdict(list)
    for key in qrels.keys():
        query_keys["_".join(key.split("_")[:-1])].append(key)

    new_results = {}
    new_qrels = {}
    for query_id_base, query_ids in query_keys.items():
        doc_scores = defaultdict(float)
        for query_id_full in query_ids:
            for doc_id, score in results[query_id_full].items():
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
                else:
                    doc_scores[doc_id] = max(score, doc_scores[doc_id])

        new_results[query_id_base] = doc_scores
        new_qrels[query_id_base] = qrels[query_id_full]  # all the same

    # now we have the new results, we can compute the scores
    _, ndcg, _map, recall, precision, naucs, mrr, naucs_mrr, cv_recall = (
        calculate_retrieval_scores(new_results, new_qrels, k_values)
    )
    score_dict = make_score_dict(
        ndcg, _map, recall, precision, naucs, mrr, naucs_mrr, cv_recall, {}
    )
    return {"max_over_subqueries_" + k: v for k, v in score_dict.items()}


def calculate_retrieval_scores(
    results: Mapping[str, Mapping[str, float]],
    qrels: RelevantDocumentsType,
    k_values: list[int],
    skip_first_result: bool = False,
) -> RetrievalEvaluationResult:
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores: dict[str, dict[str, float]] = evaluator.evaluate(results)

    (
        ndcg,
        _map,
        recall,
        precision,
        all_ndcgs,
        all_aps,
        all_recalls,
        all_precisions,
    ) = parse_metrics_from_scores(scores, k_values)
    mrr_scores = mrr(qrels, results, k_values)

    naucs = evaluate_abstention(
        results, {**all_ndcgs, **all_aps, **all_recalls, **all_precisions}
    )
    naucs_mrr = evaluate_abstention(results, mrr_scores)
    cv_recall = calculate_cv_recall(results, qrels, k_values, skip_first_result)

    avg_mrr = {k: sum(mrr_scores[k]) / len(mrr_scores[k]) for k in mrr_scores.keys()}
    return RetrievalEvaluationResult(
        all_scores=scores,
        ndcg=ndcg,
        map=_map,
        recall=recall,
        precision=precision,
        naucs=naucs,
        mrr=avg_mrr,
        naucs_mrr=naucs_mrr,
        cv_recall=cv_recall,
    )


def evaluate_abstention(
    results: Mapping[str, Mapping[str, float]],
    metric_scores: dict[str, list[float]],
) -> dict[str, float]:
    """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997

    Args:
        results: A mapping from query IDs to a dictionary of document IDs and their scores.
        metric_scores: A dictionary mapping metric names to lists of scores for each query.

    Returns:
        A dictionary mapping metric names to their corresponding nAUC scores.
    """
    all_sim_scores = [list(results[qid].values()) for qid in list(results.keys())]
    all_conf_scores = [confidence_scores(sim_scores) for sim_scores in all_sim_scores]
    conf_fcts = list(all_conf_scores[0].keys())
    all_conf_scores_ = {
        fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
    }
    metric_scores_ = {k: np.array(v) for k, v in metric_scores.items()}
    naucs = {}

    for metric_name, scores in metric_scores_.items():
        for fct, conf_scores in all_conf_scores_.items():
            naucs[f"nAUC_{metric_name}_{fct}"] = nauc(conf_scores, scores)

    return naucs


def calculate_cv_recall(
    results: Mapping[str, Mapping[str, float]],
    qrels: RelevantDocumentsType,
    k_values: list[int],
    skip_first_result: bool = False,
) -> dict[str, float]:
    """Calculate Cross-Validation Recall (CV Recall) for a set of search results.

    This function computes a binary recall-like metric at various cutoff levels (k-values).
    For each query, it checks whether at least one relevant document appears within the top-k
    retrieved results. The final score is averaged over all queries.

    Arguments:
        results: A mapping from query IDs to a dictionary of document IDs and their scores.
        qrels: A mapping from query IDs to relevant documents with relevance scores.
        k_values: A list of cutoff values at which to compute CV Recall, e.g., [1, 5, 10].
        skip_first_result: Whether to skip the top-ranked result.

    Returns:
        A dictionary mapping metric names (e.g., "CV_Recall@1") to their corresponding
        averaged scores across all queries, rounded to 5 decimal places.
    """
    all_cv_recalls = defaultdict(list)
    sorted_results: dict[str, list[tuple[str, float]]] = {
        qid: sorted(rels.items(), key=lambda item: item[1], reverse=True)
        for qid, rels in results.items()
    }

    if skip_first_result:
        for qid, rels in sorted_results.items():
            sorted_results[qid].pop(0)

    for query_id in results.keys():
        top_docs = [
            doc_id for doc_id, _ in sorted_results[query_id]
        ]  # Sorted list of doc IDs

        relevant_docs = {
            key for key in qrels.get(query_id, {}).keys() if qrels[query_id][key] != 0
        }

        for k in k_values:
            top_k_docs = top_docs[:k]

            if relevant_docs.intersection(top_k_docs):
                all_cv_recalls[k].append(1.0)
            else:
                all_cv_recalls[k].append(0.0)

    return {
        f"CV_Recall@{k}": round(sum(all_cv_recalls[k]) / len(results), 5)
        for k in k_values
    }
