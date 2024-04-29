from __future__ import annotations

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from time import time

from mteb.abstasks.TaskMetadata import TaskMetadata


logger = logging.getLogger(__name__)
from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Retrieval.fra.FQuADRetrieval import FQuADRetrieval


class FQuADRetrievalAbstention(AbsTaskAbstention, FQuADRetrieval):

    _EVAL_SPLITS = ["test", "validation"]

    metadata = TaskMetadata(
        name="FQuADRetrievalAbstention",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2019-11-01", "2020-05-01"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{dhoffschmidt-etal-2020-fquad,
    title = "{FQ}u{AD}: {F}rench Question Answering Dataset",
    author = "d{'}Hoffschmidt, Martin  and
      Belblidia, Wacim  and
      Heinrich, Quentin  and
      Brendl{\'e}, Tom  and
      Vidal, Maxime",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.107",
    doi = "10.18653/v1/2020.findings-emnlp.107",
    pages = "1193--1208",
}""",
        n_samples={"test": 400, "validation": 100},
        avg_character_length={"test": 937, "validation": 930},
    )

    def _evaluate_monolingual(
        self, retriever, corpus, queries, relevant_docs, lang=None, **kwargs
    ):
        """Function to override"""
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

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        abstention = self.abstention_evaluator.compute_abstention_scores_retrieval(
            relevant_docs, results
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **abstention,
        }
        return scores
