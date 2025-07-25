from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sentence_transformers.quantization import quantize_embeddings
from sklearn.metrics import average_precision_score

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ....abstasks.TaskMetadata import TaskMetadata

HF_REPO = "datalyes/DAPFAM_patent"
REFERENCE = "https://arxiv.org/abs/2506.22141"
BIBTEX = r"""@misc{ayaou2025dapfam,
  title        = {DAPFAM: A Domain-Aware Patent Retrieval Dataset Aggregated at the Family Level},
  author       = {Ayaou, Iliass and Cavallucci, Denis and Chibane, Hicham},
  year         = {2025},
  eprint       = {2506.22141},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}"""

_SHARED_METADATA = dict(
    dataset={"path": HF_REPO, "revision": "main"},
    reference=REFERENCE,
    type="Retrieval",
    category="p2p",
    task_subtypes=["Article retrieval", "Patent retrieval"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg@10",
    date=("1964-06-26", "2023-06-20"),  # dataset card coverage
    domains=["Engineering", "Chemistry", "Legal"],
    license="cc-by-nc-sa-4.0",
    annotations_creators="derived",
    sample_creation="created",
    bibtex_citation=BIBTEX,
)

# text-field dictionaries
_QUERY_FIELDS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
}
_CORPUS_FIELDS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
    "TitleAbstractClaimsDescription": [
        "title_en",
        "abstract_en",
        "claims_text",
        "description_en",
    ],
}

# paper variants used in Table 4
_IN_PAPER = {
    ("TitleAbstract", "TitleAbstractClaims"),
    ("TitleAbstractClaims", "TitleAbstractClaims"),
}


# ───────────────────────────────────────────────────
# MIX-IN with shared logic + metric implementation
class _DAPFAMMixin:
    # class-level attributes are filled in each concrete subclass
    domain_filter: Optional[str] = None
    query_fields: List[str] = []
    corpus_fields: List[str] = []
    in_paper: bool = False

    def load_data(self, **_) -> Tuple[Dict, Dict, Dict]:
        ds_c = load_dataset(HF_REPO, "corpus", split="train")
        ds_q = load_dataset(HF_REPO, "queries", split="train")
        ds_r = load_dataset(HF_REPO, "relations", split="train")

        self.corpus = {
            "test": {
                r["relevant_id"]: "\n".join(
                    str(r[f]) for f in self.corpus_fields if r.get(f)
                )
                for r in ds_c
            }
        }
        self.queries = {
            "test": {
                r["query_id"]: "\n".join(
                    str(r[f]) for f in self.query_fields if r.get(f)
                )
                for r in ds_q
            }
        }

        qrels: Dict[str, Dict[str, Tuple[float, str]]] = {}
        for r in ds_r:
            qid, pid = r["query_id"], r["relevant_id"]
            qrels.setdefault(qid, {})[pid] = (
                float(r["relevance_score"]),
                r["domain_rel"],
            )
        # 4) Assign for MTEB
        self.relevant_docs = {"test": qrels}
        self.data_loaded = True
        return self.corpus, self.queries, self.relevant_docs

    # ------------ evaluation  (faithful to the paper) ------------
    def _dapfam_evaluate(
        self,
        model_wrapper,
        split: str = "test",
        subsets_to_run=None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Custom evaluation that quantises embeddings to uint8 before
        normalisation (per the paper) and
        computes recall / nDCG / mAP exactly like the paper.
        It is fully deterministic.
        """
        if not getattr(self, "data_loaded", False):
            self.load_data()

        corpus = self.corpus[split]
        queries = self.queries[split]
        qrels = self.relevant_docs[split]

        corp_ids, corp_texts = zip(*corpus.items())
        qry_ids, qry_texts = zip(*queries.items())

        encode_kwargs = kwargs.get("encode_kwargs", {})
        emb_c = model_wrapper.model.encode(
            list(corp_texts), **encode_kwargs, show_progress_bar=True
        )
        emb_q = model_wrapper.model.encode(
            list(qry_texts), **encode_kwargs, show_progress_bar=True
        )

        # uint8 quantisation (per paper)
        emb_c = quantize_embeddings(emb_c, precision="uint8")
        emb_q = quantize_embeddings(emb_q, precision="uint8")

        # cosine similarity
        emb_c = emb_c / np.linalg.norm(emb_c, axis=1, keepdims=True)
        emb_q = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True)
        sims = emb_q @ emb_c.T

        # ranking per query Dict[str, List[str]]
        run: Dict[str, List[str]] = {}
        for i, qid in enumerate(qry_ids):
            scores = sims[i]
            idxs = np.argsort(-scores)
            run[qid] = [(corp_ids[j], float(scores[j])) for j in idxs]

        # ---- metric helpers ----
        def ndcg_at_k(preds: List[str], refset: set[str], k: int) -> float:
            if not refset:
                return 1.0
            gains = [1.0 if pid in refset else 0.0 for pid in preds[:k]]

            def dcg(g):
                return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(g))

            ideal = sorted(gains, reverse=True)
            idcg = dcg(ideal)
            # if ideal DCG is zero, return zero per paper
            if idcg <= 0.0:
                return 0.0
            return dcg(gains) / idcg

        rec10 = []
        rec100 = []
        ndc10 = []
        ndc100 = []
        map10 = []
        map100 = []

        for qid, ranking in run.items():
            preds = [pid for pid, _ in ranking]
            full = {d for d, (s, _) in qrels.get(qid, {}).items() if s > 0}
            if self.domain_filter:
                relset = {
                    pid
                    for pid, (s, dom) in qrels[qid].items()
                    if s > 0 and dom == self.domain_filter
                }
            else:
                relset = full

            # recall@K
            for k, rec_list in ((10, rec10), (100, rec100)):
                hits = len(set(preds[:k]) & relset)
                rec = hits / len(relset) if relset else 1.0
                rec_list.append(rec)

            # nDCG@K
            ndc10.append(ndcg_at_k(preds, relset, 10))
            ndc100.append(ndcg_at_k(preds, relset, 100))

            # mAP@K via rank-based scores over *top-K* only, per paper
            for k, map_list in ((10, map10), (100, map100)):
                # build binary truth for the top-k
                topk = preds[:k]
                y_true = [1 if pid in relset else 0 for pid in topk]
                # if no positives exist, perfect; else zero if none in top-k
                if sum(y_true) == 0:
                    ap = 1.0 if not relset else 0.0
                else:
                    # rank‐based scores  k, k−1, …, 1
                    y_scores = [k - i for i in range(k)]
                    ap = average_precision_score(y_true, y_scores)
                map_list.append(ap)

        return {
            "default": {
                "recall@10": float(np.mean(rec10)),
                "recall@100": float(np.mean(rec100)),
                "ndcg@10": float(np.mean(ndc10)),
                "ndcg@100": float(np.mean(ndc100)),
                "map@10": float(np.mean(map10)),
                "map@100": float(np.mean(map100)),
                "main_score": float(np.mean(ndc10)),
            }
        }

    def evaluate(
        self,
        model_wrapper,
        split: str = "test",
        subsets_to_run=None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        return self._dapfam_evaluate(model_wrapper, split, subsets_to_run, **kwargs)


# ───────────────────────────────────────────────────
# helper to build TaskMetadata
def _meta(name: str, desc: str) -> TaskMetadata:
    return TaskMetadata(name=name, description=desc, **_DEFAULT_META)


# ───────────────────────────────────────────────────
# 18 explicit task classes (no loops)

# NOTE: Each class only sets class-level attributes

# AbsTaskRetrieval.__init__ will call self.load_data() and compute metadata automatically.


# ---------- ALL domain ----------
class Dapfam_ALL_TitleAbstract_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "ALL • Query: TA  | Corpus: TA")


class Dapfam_ALL_TitleAbstract_TitleAbstractClaims(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "ALL • Query: TA  | Corpus: TA+Claims  (paper)")


class Dapfam_ALL_TitleAbstract_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "ALL • Query: TA  | Corpus: TA+Claims+Desc")


class Dapfam_ALL_TitleAbstractClaims_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "ALL • Query: TA+Claims | Corpus: TA")


class Dapfam_ALL_TitleAbstractClaims_TitleAbstractClaims(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "ALL • Query: TA+Claims | Corpus: TA+Claims (paper)")


class Dapfam_ALL_TitleAbstractClaims_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "ALL • Query: TA+Claims | Corpus: TA+Claims+Desc")


# ---------- IN domain ----------
class Dapfam_IN_TitleAbstract_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "IN • Query: TA  | Corpus: TA")


class Dapfam_IN_TitleAbstract_TitleAbstractClaims(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "IN • Query: TA  | Corpus: TA+Claims  (paper)")


class Dapfam_IN_TitleAbstract_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "IN • Query: TA  | Corpus: TA+Claims+Desc")


class Dapfam_IN_TitleAbstractClaims_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "IN • Query: TA+Claims | Corpus: TA")


class Dapfam_IN_TitleAbstractClaims_TitleAbstractClaims(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "IN • Query: TA+Claims | Corpus: TA+Claims (paper)")


class Dapfam_IN_TitleAbstractClaims_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "IN • Query: TA+Claims | Corpus: TA+Claims+Desc")


# ---------- OUT domain ----------
class Dapfam_OUT_TitleAbstract_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "OUT • Query: TA  | Corpus: TA")


class Dapfam_OUT_TitleAbstract_TitleAbstractClaims(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "OUT • Query: TA  | Corpus: TA+Claims  (paper)")


class Dapfam_OUT_TitleAbstract_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "OUT • Query: TA  | Corpus: TA+Claims+Desc")


class Dapfam_OUT_TitleAbstractClaims_TitleAbstract(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = _meta(__qualname__, "OUT • Query: TA+Claims | Corpus: TA")


class Dapfam_OUT_TitleAbstractClaims_TitleAbstractClaims(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = _meta(__qualname__, "OUT • Query: TA+Claims | Corpus: TA+Claims (paper)")


class Dapfam_OUT_TitleAbstractClaims_TitleAbstractClaimsDescription(
    _DAPFAMMixin, AbsTaskRetrieval
):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = _meta(__qualname__, "OUT • Query: TA+Claims | Corpus: TA+Claims+Desc")
