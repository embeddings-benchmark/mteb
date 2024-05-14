from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import pytrec_eval

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class AbstentionEvaluator(Evaluator):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __call__(self, model):
        """This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        raise NotImplementedError("The abstention evaluator must not be called directly.")

    @staticmethod
    def compute_abstention_scores_retrieval(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int] = [1, 3, 5, 10],
        ignore_identical_ids: bool = True
    ) -> Dict[str, float]:

        # Choose whether to ignore identical ids
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        # Compute retrieval metrics for each instance
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)
      
        # Compute confidence scores (max, std, 1-2)
        conf_scores = {}
        for qid in results.keys():
            scs = list(results[qid].values())
            scs_mean = sum(scs) / len(scs)
            scs_sort = sorted(scs)[::-1]
            conf_scores[qid] = {
                'max': scs_sort[0], 
                'std': (sum((sc - scs_mean) ** 2 for sc in scs) / len(scs)) ** (1/2), 
                '1-2': scs_sort[0] - scs_sort[1]
            }
        
        # Compute nAUCs
        metrics = list(list(scores.values())[0].keys())
        abst_funcs = list(list(conf_scores.values())[0].keys())
        abst_rates = [k/10 for k in range(10)]
        abst_scores = {}

        # Evaluate for all abstention functions (max, std, 1-2)
        for abst_func in abst_funcs:
            conf_scs = {key: val[abst_func] for key, val in conf_scores.items()}
            conf_scs_sort = dict(sorted(conf_scs.items(), key=lambda item: item[1])[::-1])
            
            evals = {metric: [] for metric in metrics}
            oracles = {metric: [] for metric in metrics}

            # Evaluate for all abstention rates
            for abst_rate in abst_rates:
                num_kept = len(conf_scs) - int(abst_rate * len(conf_scs))
                kept_qids = list(conf_scs_sort.keys())[:num_kept]

                # Evaluate for all metrics (ndcg, map, precision, recall)
                for metric in metrics:
                    evals[metric].append(
                        sum(scores[qid][metric] for qid in kept_qids) / num_kept
                    )
                    scs = [scores[qid][metric] for qid in scores.keys()]
                    scs_sort = sorted(scs)[::-1]
                    oracles[metric].append(sum(scs_sort[:num_kept]) / num_kept)
            
            # Compute nAUCs
            for metric in metrics:
                auc = sum(evals[metric]) / len(abst_rates) * max(abst_rates)
                auc_oracle = sum(oracles[metric]) / len(abst_rates) * max(abst_rates)
                auc_rand = oracles[metric][0] * max(abst_rates)
                abst_scores[
                    f"nAUC_{metric.replace('P_', 'precision_at_').replace('recall_', 'recall_at_').replace('cut', 'at')}_{abst_func}"
                ] = (auc - auc_rand) / (auc_oracle - auc_rand)

        return abst_scores
