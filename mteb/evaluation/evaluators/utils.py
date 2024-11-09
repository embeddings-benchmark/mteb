from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pytrec_eval
import requests
import torch
import tqdm
from datasets import load_dataset
from packaging.version import Version
from sklearn.metrics import auc

try:
    # speeds up computation if available
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """Calculate pairwise cosine similarities between two sets of vectors.

    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    # since compile works better with pure tensor operations
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    # The actual function to compile
    def _cos_sim_core(a_tensor, b_tensor):
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.unsqueeze(0)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a_tensor, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b_tensor, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    # Compile the core function once
    if hasattr(torch, "compile"):  # Check if torch.compile is available
        _cos_sim_core_compiled = torch.compile(_cos_sim_core)
        return _cos_sim_core_compiled(a, b)
    else:
        return _cos_sim_core(a, b)


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    # The actual function to compile
    def _dot_score_core(a_tensor, b_tensor):
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.unsqueeze(0)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.unsqueeze(0)

        return torch.mm(a_tensor, b_tensor.transpose(0, 1))

    # Compile the core function once
    if hasattr(torch, "compile"):  # Check if torch.compile is available
        _dot_score_core_compiled = torch.compile(_dot_score_core)
        return _dot_score_core_compiled(a, b)
    else:
        return _dot_score_core(a, b)


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
    output_type: str = "mean",
) -> tuple[dict[str, float]]:
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
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
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
    k_values: list[int],
    output_type: str = "mean",
) -> tuple[dict[str, float]]:
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
    k_values: list[int],
    output_type: str = "mean",
) -> tuple[dict[str, float]]:
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
    k_values: list[int],
    output_type: str = "mean",
) -> dict[str, float]:
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
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
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
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, float]],
    task_name: str,
    k_values: list[int],
) -> dict[str, float]:
    """Computes the scores needed for FollowIR datasets, including p-MRR (measuring change in instruction) and
    details about the original instruction run and changed instruction run.
    """
    followir_scores = defaultdict(dict)
    # load the qrel_diff from the dataset
    TASK_TO_HF_DATASET = {
        "Core17InstructionRetrieval": ("jhu-clsp/core17-instructions-mteb", False),
        "Robust04InstructionRetrieval": ("jhu-clsp/robust04-instructions-mteb", False),
        "News21InstructionRetrieval": ("jhu-clsp/news21-instructions-mteb", False),
        "mFollowIR": ("jhu-clsp/mfollowir-parquet-mteb", True),
        "mFollowIRCrossLingual": (
            "jhu-clsp/mfollowir-cross-lingual-parquet-mteb",
            True,
        ),
    }
    hf_path, is_multilingual = TASK_TO_HF_DATASET[task_name]
    if is_multilingual:
        # figure out which of the languages this is: ["zho", "rus", "fas"]
        # gather the changed_qrels for each, and store the keys as a check
        for lang in ["zho", "rus", "fas"]:
            config_name = f"qrel_diff-{lang}"
            changed_qrels = {
                item["query-id"]: item["corpus-ids"]
                for item in load_dataset(hf_path, config_name)["qrel_diff"]
            }
            potential_keys = {item + "-og" for item in changed_qrels.keys()} | {
                item + "-changed" for item in changed_qrels.keys()
            }
            if (
                potential_keys == set(qrels.keys())
                or len(potential_keys - set(qrels.keys())) <= 2
            ):  # there are about two skipped
                break  # this is the right qrels

    else:
        changed_qrels = {
            item["query-id"]: item["corpus-ids"]
            for item in load_dataset(hf_path, "qrel_diff")["qrel_diff"]
        }

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
        _, ndcg, _map, recall, precision, naucs = calculate_retrieval_scores(
            group, qrels_sep[name], k_values
        )
        # add these to the followir_scores with name prefix
        scores_dict = make_score_dict(ndcg, _map, recall, precision, {}, naucs, {}, {})
        for key, value in scores_dict.items():
            followir_scores[name][key] = value

    return followir_scores


def rank_score(x: dict[str, float]) -> float:
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    else:
        return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))


# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm.tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def convert_conv_history_to_query(conversations: list[list[str | dict]]) -> str:
    conversations_converted = []

    for conversation in conversations:
        # if it's a list of strings, just join them
        if isinstance(conversation[0], str):
            conv_str = "; ".join(conversation)
        # otherwise, it's a list of dictionaries, which we need to convert to strings
        elif isinstance(conversation[0], dict):
            conv = []
            for i, turn in enumerate(conversation):
                error_msg = (
                    "When converting conversations lists of dictionary to string, each turn in the conversation "
                    + "must be a dictionary with 'role' and 'content' keys"
                )
                if not isinstance(turn, dict):
                    raise ValueError(f"Turn {i} is not a dictionary. " + error_msg)

                # check for keys 'role' and 'content' in the dictionary, if not found, raise an error
                if "role" not in turn:
                    raise ValueError(
                        "Key 'role' not found in the dictionary. " + error_msg
                    )
                if "content" not in turn:
                    raise ValueError(
                        "Key 'content' not found in the dictionary. " + error_msg
                    )

                conv.append(f"{turn['role']}: {turn['content']}")
            conv_str = "; ".join(conv)
        else:
            raise ValueError(
                "Conversations must be a list consisting of strings or dictionaries with 'role' and 'content' keys"
            )

        conversations_converted.append(conv_str)

    return conversations_converted


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
        # argsort stable=True is default in numpy >2.0.0
        if Version(np.__version__) < Version("2.0.0"):
            conf_scores_argsort = np.argsort(conf_scores)
        else:
            conf_scores_argsort = np.argsort(conf_scores, stable=True)
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


def add_task_specific_scores(
    scores: dict[str, float],
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    task_name: str,
    k_values: list[int],
) -> dict[str, float]:
    """Add task-specific scores to the scores dictionary, that are not needed for all results but require additional computation."""
    task_scores = {}
    if task_name in ["NevIR"]:
        paired_score = paired_accuracy(qrels, results, scores)
        task_scores["paired_accuracy"] = paired_score

    if task_name in ["InstructIR"]:
        robustness_at_10_score = robustness_at_10(qrels, results, scores)
        task_scores["robustness_at_10"] = robustness_at_10_score

    if task_name in [
        "mFollowIR",
        "mFollowIRCrossLingual",
        "Robust04InstructionRetrieval",
        "Core17InstructionRetrieval",
        "News21InstructionRetrieval",
    ]:
        p_mrr_and_consolidated_scores = evaluate_p_mrr_change(
            results, qrels, task_name, k_values
        )
        task_scores.update(p_mrr_and_consolidated_scores)

    if task_name in ["MindSmallReranking"]:
        take_max_over_subqueries = max_over_subqueries(qrels, results, k_values)
        task_scores.update(take_max_over_subqueries)

    return task_scores


def paired_accuracy(
    qrels: dict[str, dict[str, float]],
    results: dict[str, dict[str, float]],
    scores: dict[str, float],
) -> float:
    """Computes the paired accuracy. This means both queries for an instance have to be correct for it to count.
        This is because models will prefer one passage all the time, giving it 50% automatically unless we correct for this.
        For more details, see https://arxiv.org/abs/2305.07614

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        scores: The scores for the queries, to extract top_1 recall for each query
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
    qrels: dict[str, dict[str, float]],
    results: dict[str, dict[str, float]],
    scores: dict[str, float],
) -> float:
    """Computes the robustness at 10. This computes the lowest ndcg@10 over all instructions. Taken from https://arxiv.org/abs/2402.14334

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        scores: The scores for the queries, to extract ndcg@10 for each query
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


def make_score_dict(ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, task_scores):
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
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
        **task_scores,
    }
    return scores


def parse_metrics_from_scores(scores, k_values):
    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}
    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)

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


def max_over_subqueries(qrels, results, k_values):
    """Computes the max over subqueries scores when merging.

    Args:
        qrels: Ground truth relevance judgments for the queries
        results: Predicted relevance scores for the queries
        k_values: The k values for which to compute the scores
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
    _, ndcg, _map, recall, precision, naucs = calculate_retrieval_scores(
        new_results, new_qrels, k_values
    )
    score_dict = make_score_dict(ndcg, _map, recall, precision, {}, naucs, {}, {})
    return {"max_over_subqueries_" + k: v for k, v in score_dict.items()}


def calculate_retrieval_scores(results, qrels, k_values):
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

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

    naucs = evaluate_abstention(
        results, {**all_ndcgs, **all_aps, **all_recalls, **all_precisions}
    )

    return scores, ndcg, _map, recall, precision, naucs


def evaluate_abstention(
    results: dict[str, dict[str, float]],
    metric_scores: dict[str, list[float]],
) -> dict[str, float]:
    """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997"""
    all_sim_scores = [list(results[qid].values()) for qid in list(results.keys())]
    all_conf_scores = [confidence_scores(sim_scores) for sim_scores in all_sim_scores]
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
