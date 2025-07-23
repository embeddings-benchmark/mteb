from __future__ import annotations

import math

import numpy as np
from datasets import load_dataset
from sentence_transformers.quantization import quantize_embeddings
from sklearn.metrics import average_precision_score

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ....abstasks.TaskMetadata import TaskMetadata
from ....load_results.task_results import TaskResult

HF_REPO = "datalyes/DAPFAM_patent"
REFERENCE = "https://arxiv.org/abs/2506.22141"
BIBTEX = r"""@article{ayaou2025dapfam,
  title={DAPFAM: A Domain-Aware Patent Retrieval Dataset Aggregated at the Family Level},
  author={Ayaou, Iliass and Cavallucci, Denis and Chibane, Hicham},
  journal={arXiv preprint arXiv:2506.22141},
  year={2025}
}"""


DOMAIN_LABELS = {"ALL": None, "IN": "IN", "OUT": "OUT"}


QUERY_VARIANTS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
}
CORPUS_VARIANTS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
    "TitleAbstractClaimsDescription": [
        "title_en",
        "abstract_en",
        "claims_text",
        "description_en",
    ],
}


def make_load_data(q_fields, c_fields):
    def load_data(self, **kwargs):
        # 1) Pull HF splits
        ds_c = load_dataset(HF_REPO, "corpus", split="train")
        ds_q = load_dataset(HF_REPO, "queries", split="train")
        ds_r = load_dataset(HF_REPO, "relations", split="train")
        # 2) Build dicts
        corpus = {
            r["relevant_id"]: "\n".join(
                str(r[f]) for f in c_fields if r.get(f) is not None
            )
            for r in ds_c
        }
        queries = {
            r["query_id"]: "\n".join(
                str(r[f]) for f in q_fields if r.get(f) is not None
            )
            for r in ds_q
        }

        qrels = {}
        for r in ds_r:
            qid, pid = r["query_id"], r["relevant_id"]
            qrels.setdefault(qid, {})[pid] = (
                float(r["relevance_score"]),
                r["domain_rel"],
            )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": qrels}
        self.data_loaded = True
        return self.corpus, self.queries, self.relevant_docs

    return load_data


def make_evaluate(domain_filter):
    def evaluate(
        self, model_wrapper, split="test", subsets_to_run=None, **kwargs
    ) -> TaskResult:
        if not getattr(self, "data_loaded", False):
            self.load_data()
        corpus = self.corpus[split]
        queries = self.queries[split]
        qrels_map = self.relevant_docs[split]

        encode_kwargs = kwargs.get("encode_kwargs", {})
        corp_ids, corp_txts = zip(*corpus.items())
        qry_ids, qry_txts = zip(*queries.items())

        emb_c = model_wrapper.model.encode(
            list(corp_txts), **encode_kwargs, show_progress_bar=True
        )
        emb_q = model_wrapper.model.encode(
            list(qry_txts), **encode_kwargs, show_progress_bar=True
        )

        # Quantize the embeddings
        emb_c = quantize_embeddings(emb_c, precision="uint8")
        emb_q = quantize_embeddings(emb_q, precision="uint8")

        emb_c = emb_c / np.linalg.norm(emb_c, axis=1, keepdims=True)
        emb_q = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True)

        sims = emb_q.dot(emb_c.T)

        run_dict = {}
        for i, qid in enumerate(qry_ids):
            scores = sims[i]
            idxs = np.argsort(-scores)
            run_dict[qid] = [(corp_ids[j], float(scores[j])) for j in idxs]

        def ndcg_at_k(preds, refset, k):
            if not refset:
                return 1.0
            gains = [1.0 if pid in refset else 0.0 for pid in preds[:k]]

            def dcg(g):
                return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(g))

            ideal = sorted(gains, reverse=True)
            idcg = dcg(ideal)

            if idcg <= 0.0:
                return 0.0
            return dcg(gains) / idcg

        rec10 = []
        rec100 = []
        ndc10 = []
        ndc100 = []
        map10 = []
        map100 = []
        for qid, ranking in run_dict.items():
            preds = [pid for pid, _ in ranking]

            full = {pid for pid, (s, _) in qrels_map.get(qid, {}).items() if s > 0}
            if domain_filter:
                relset = {
                    pid
                    for pid, (s, dom) in qrels_map[qid].items()
                    if s > 0 and dom == domain_filter
                }
            else:
                relset = full

            for k, rec_list in ((10, rec10), (100, rec100)):
                hits = len(set(preds[:k]) & relset)
                rec = hits / len(relset) if relset else 1.0
                rec_list.append(rec)

            ndc10.append(ndcg_at_k(preds, relset, 10))
            ndc100.append(ndcg_at_k(preds, relset, 100))

            for k, map_list in ((10, map10), (100, map100)):
                topk = preds[:k]
                y_true = [1 if pid in relset else 0 for pid in topk]
                if sum(y_true) == 0:
                    ap = 1.0 if not relset else 0.0
                else:
                    y_scores = [k - i for i in range(k)]
                    ap = average_precision_score(y_true, y_scores)
                map_list.append(ap)

        # 6) aggregate macro-averages
        metrics = {
            "recall@10": float(np.mean(rec10)),
            "recall@100": float(np.mean(rec100)),
            "ndcg@10": float(np.mean(ndc10)),
            "ndcg@100": float(np.mean(ndc100)),
            "map@10": float(np.mean(map10)),
            "map@100": float(np.mean(map100)),
            "main_score": float(np.mean(ndc10)),
        }

        return {"default": metrics}

    return evaluate


# ——— register all tasks ———
for domain, domlbl in DOMAIN_LABELS.items():
    for qn, qf in QUERY_VARIANTS.items():
        for cn, cf in CORPUS_VARIANTS.items():
            task_name = f"Dapfam_{domain}_{qn}_{cn}"
            metadata = TaskMetadata(
                name=task_name,
                description=f"DAPFAM [{domain}] Q={qn} / C={cn}",
                dataset={"path": HF_REPO, "revision": "main"},
                reference=REFERENCE,
                type="Retrieval",
                category="p2p",
                task_subtypes=["Patent retrieval", "Article retrieval"],
                eval_splits=["test"],
                eval_langs=["eng-Latn"],
                main_score="ndcg@10",
                date=("2025-06-30", "2025-06-30"),
                domains=["Engineering", "Chemistry", "Legal"],
                license="not specified",
                annotations_creators="derived",
                sample_creation="created",
                judged_docs_only_flag=False,
                bibtex_citation=BIBTEX,
            )

            def __init__(self):
                super(self.__class__, self).__init__()
                self.load_data()
                self.calculate_metadata_metrics()

            attrs = {
                "__init__": __init__,
                "metadata": metadata,
                "load_data": make_load_data(qf, cf),
                "evaluate": make_evaluate(domlbl),
            }
            globals()[task_name] = type(task_name, (AbsTaskRetrieval,), attrs)
