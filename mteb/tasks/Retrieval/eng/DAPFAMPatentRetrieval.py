from __future__ import annotations

import math

import numpy as np
from datasets import load_dataset
from sentence_transformers.quantization import quantize_embeddings
from sklearn.metrics import average_precision_score

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval, logger
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
    dataset={"path": HF_REPO, "revision": "3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544"},
    reference=REFERENCE,
    type="Retrieval",
    category="p2p",
    task_subtypes=["Article retrieval", "Patent retrieval"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg@10",
    date=("1964-06-26", "2023-06-20"),
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


# MIX-IN with shared logic + metric implementation
class _DAPFAMMixin:
    # class-level attributes are filled in each concrete subclass
    domain_filter: str | None = None
    query_fields: list[str] = []
    corpus_fields: list[str] = []
    in_paper: bool = False

    def load_data(self, **_) -> tuple[dict, dict, dict]:
        ds_c = load_dataset(HF_REPO, "corpus", split="train",revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544")
        ds_q = load_dataset(HF_REPO, "queries", split="train", revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544")
        ds_r = load_dataset(HF_REPO, "relations", split="train", revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544")

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

        qrels: dict[str, dict[str, tuple[float, str]]] = {}
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

    def _dapfam_evaluate(
        self,
        model_wrapper,
        split: str = "test",
        subsets_to_run=None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Custom evaluation that quantises embeddings to uint8 before
        normalisation (per the paper) and
        computes recall / nDCG / mAP exactly like the paper if quantize=True and similarity=cosine.
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
        quantize = kwargs.get("quantize", False)

        # check similarity function name :
        logger.info(model_wrapper.model.similarity_fn_name)
        emb_c = model_wrapper.model.encode(
            list(corp_texts), **encode_kwargs, show_progress_bar=True
        )
        emb_q = model_wrapper.model.encode(
            list(qry_texts), **encode_kwargs, show_progress_bar=True
        )

        # uint8 quantisation (per paper) if chosen then we go back to fp32 to avoid error
        # by sentence transformers similarity function (doesn't accept quantized embeddings)
        if quantize:
            emb_c_q = quantize_embeddings(emb_c, precision="uint8")
            emb_q_q = quantize_embeddings(emb_q, precision="uint8")
            emb_c = emb_c_q.astype(np.float32)
            emb_q = emb_q_q.astype(np.float32)

        sims = model_wrapper.model.similarity(emb_q, emb_c).cpu().numpy()

        # ranking per query dict[str, list[str]]
        run: dict[str, list[str]] = {}
        for i, qid in enumerate(qry_ids):
            scores = sims[i]
            idxs = np.argsort(-scores)
            run[qid] = [(corp_ids[j], float(scores[j])) for j in idxs]

        # ---- metric helpers ----
        def ndcg_at_k(preds: list[str], refset: set[str], k: int) -> float:
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
    ) -> dict[str, dict[str, float]]:
        return self._dapfam_evaluate(model_wrapper, split, subsets_to_run, **kwargs)


# ───────────────────────────────────────────────────
# DAPFAM Patent Family Retrieval Tasks


class DAPFAMAllTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )
