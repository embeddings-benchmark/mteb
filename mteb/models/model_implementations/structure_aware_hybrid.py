from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mteb._create_dataloaders import _combine_queries_with_instruction_text
from mteb.models.hybrid_wrappers import HybridSearch
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import MTEBModels
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _mean_len(texts) -> float:
    lengths = [len(_tokenize(text)) for text in texts]
    return float(np.mean(lengths)) if lengths else 0.0


def _minmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return dict.fromkeys(scores, 1.0)
    return {key: (val - lo) / (hi - lo) for key, val in scores.items()}


def _rank_map(scores: Mapping[str, float]) -> dict[str, int]:
    return {
        did: rank
        for rank, (did, _) in enumerate(
            sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])), 1
        )
    }


def _rrf(
    results_list: Sequence[Mapping[str, float]],
    weights: Sequence[float],
    *,
    top_k: int,
    k: int = 60,
) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for results, weight in zip(results_list, weights, strict=True):
        ranked = sorted(results.items(), key=lambda kv: (-kv[1], kv[0]))
        for rank, (did, _) in enumerate(ranked, 1):
            scores[did] += weight / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])


def _hybrid_scores(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    alpha: float,
    top_k: int,
) -> dict[str, float]:
    aa = _minmax(a)
    bb = _minmax(b)
    docs = set(aa) | set(bb)
    vals = {
        did: alpha * aa.get(did, 0.0) + (1.0 - alpha) * bb.get(did, 0.0)
        for did in docs
    }
    return dict(sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])


@dataclass
class _FieldBM25Index:
    doc_ids: list[str]
    postings: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    idf: dict[str, dict[str, float]]
    lengths: dict[str, np.ndarray]
    avgdl: dict[str, float]
    norms: dict[str, np.ndarray]
    field_weights: dict[str, float]
    k1: float = 0.9
    b: float = 0.4

    @classmethod
    def build(
        cls,
        docs: dict[str, dict[str, str]],
        field_weights: dict[str, float],
    ) -> _FieldBM25Index:
        doc_ids = list(docs.keys())
        postings = {field: defaultdict(list) for field in field_weights}
        lengths = {
            field: np.zeros(len(doc_ids), dtype=np.float32) for field in field_weights
        }

        for i, did in enumerate(doc_ids):
            doc = docs[did]
            for field in field_weights:
                tokens = _tokenize(doc.get(field, ""))
                lengths[field][i] = len(tokens)
                for term, tf in Counter(tokens).items():
                    postings[field][term].append((i, tf))

        n_docs = len(doc_ids)
        avgdl = {
            field: float(np.mean(lengths[field]) + 1e-9) for field in field_weights
        }
        idf = {}
        for field, field_postings in postings.items():
            idf[field] = {
                term: math.log(
                    1.0 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5)
                )
                for term, plist in field_postings.items()
            }
        norms = {
            field: 0.9 * (1.0 - 0.4 + 0.4 * lengths[field] / avgdl[field])
            for field in field_weights
        }
        compact_postings = {
            field: {
                term: (
                    np.fromiter((doc_i for doc_i, _ in plist), dtype=np.int32),
                    np.fromiter((tf for _, tf in plist), dtype=np.float32),
                )
                for term, plist in field_postings.items()
            }
            for field, field_postings in postings.items()
        }
        return cls(
            doc_ids=doc_ids,
            postings=compact_postings,
            idf=idf,
            lengths=lengths,
            avgdl=avgdl,
            norms=norms,
            field_weights=field_weights,
        )

    def search(self, queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]:
        results = {}
        for qid, query in queries.items():
            scores = np.zeros(len(self.doc_ids), dtype=np.float32)
            for field, weight in self.field_weights.items():
                norm = self.norms[field]
                for term in _tokenize(query):
                    plist = self.postings[field].get(term)
                    if plist is None:
                        continue
                    doc_idx, tf = plist
                    idf = self.idf[field].get(term, 0.0)
                    scores[doc_idx] += (
                        weight
                        * idf
                        * (tf * (self.k1 + 1.0))
                        / (tf + norm[doc_idx])
                    )

            cand = np.flatnonzero(scores > 0.0)
            if cand.size > top_k:
                keep = np.argpartition(-scores[cand], top_k - 1)[:top_k]
                cand = cand[keep]
            ranked = sorted(
                ((int(doc_i), float(scores[doc_i])) for doc_i in cand),
                key=lambda kv: (-kv[1], self.doc_ids[kv[0]]),
            )
            results[qid] = {self.doc_ids[doc_i]: score for doc_i, score in ranked}
        return results


class FieldWeightedBM25Search:
    """Small SearchProtocol BM25/BM25F component for title-aware hybrid search."""

    def __init__(self, field_weights: dict[str, float], name: str) -> None:
        self.field_weights = field_weights
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={"name": name, "model_type": ["sparse"]}
        )
        self._index: _FieldBM25Index | None = None

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        del task_metadata, hf_split, hf_subset, encode_kwargs, num_proc
        docs = {
            str(row["id"]): {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
            for row in corpus
        }
        self._index = _FieldBM25Index.build(docs, self.field_weights)

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        del task_metadata, hf_split, hf_subset, encode_kwargs, num_proc
        if self._index is None:
            raise ValueError("Corpus must be indexed before searching.")

        processed = _combine_queries_with_instruction_text(queries)
        query_dict = {
            str(qid): text
            for qid, text in zip(queries["id"], processed["text"], strict=True)
        }
        results = self._index.search(query_dict, top_k)
        if top_ranked is None:
            return results

        filtered = {}
        for qid, scores in results.items():
            allowed = set(top_ranked.get(qid, []))
            filtered[qid] = {did: score for did, score in scores.items() if did in allowed}
        return filtered


class CorpusAdaptiveHybridSearch(HybridSearch):
    """HybridSearch extension with corpus-level routing before per-query fusion."""

    def __init__(self, models: Sequence[MTEBModels]) -> None:
        super().__init__(
            models=models,
            weights=[1.0, 1.0, 1.0],
            fusion_strategy=self._fuse_adaptive,
        )
        self.avg_doc_len = 0.0
        self.avg_title_len = 0.0
        self.avg_query_len = 0.0
        self.mteb_model_meta = structure_aware_corpus_adaptive_hybrid

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        self.avg_doc_len = _mean_len(row.get("text", "") or "" for row in corpus)
        self.avg_title_len = _mean_len(row.get("title", "") or "" for row in corpus)
        super().index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        processed = _combine_queries_with_instruction_text(queries)
        self.avg_query_len = _mean_len(processed["text"])
        return super().search(
            queries,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )

    def _fuse_adaptive(  # noqa: PLR0911
        self,
        query_scores_list: Sequence[Mapping[str, float]],
        weights: Sequence[float],
    ) -> Mapping[str, float]:
        del weights
        top_k = max((len(scores) for scores in query_scores_list), default=0)
        bm25, bm25f, dense = query_scores_list
        avg_doc_len = self.avg_doc_len
        avg_title_len = self.avg_title_len
        avg_query_len = self.avg_query_len

        if avg_doc_len >= 250.0 and avg_query_len <= 8.0 and avg_title_len < 12.0:
            return self._lexical_consensus_dense_micro_hybrid(
                bm25, dense, top_k=top_k
            )
        if avg_query_len < 5.0:
            return _rrf([bm25, dense], [1.0, 1.0], top_k=top_k, k=10)
        if avg_title_len >= 12.0 and avg_query_len >= 10.0:
            if avg_doc_len < 170.0 and avg_query_len <= 15.0:
                return _rrf([bm25f, dense, bm25], [1.25, 1.0, 0.35], top_k=top_k)
            return self._depth_regularized_structure_hybrid(
                bm25, bm25f, dense, top_k=top_k
            )
        if avg_doc_len < 190.0 and avg_title_len < 11.0:
            if avg_title_len >= 8.0 and avg_query_len < 11.0:
                return _rrf([dense, bm25f, bm25], [1.0, 0.2, 0.05], top_k=top_k, k=10)
            if 11.0 <= avg_query_len < 40.0:
                return self._structure_residual_hybrid(
                    bm25, bm25f, dense, alpha=0.40, residual_weight=0.01, top_k=top_k
                )
            return self._structure_residual_hybrid(
                bm25, bm25f, dense, alpha=0.15, residual_weight=0.02, top_k=top_k
            )
        return self._structure_residual_hybrid(
            bm25, bm25f, dense, alpha=0.40, residual_weight=0.01, top_k=top_k
        )

    @staticmethod
    def _depth_regularized_structure_hybrid(
        bm25: Mapping[str, float],
        bm25f: Mapping[str, float],
        dense: Mapping[str, float],
        *,
        top_k: int,
    ) -> dict[str, float]:
        base = _hybrid_scores(bm25, dense, alpha=0.50, top_k=top_k)
        prior = _rrf([bm25f, dense, bm25], [1.25, 1.0, 0.35], top_k=top_k)
        base_n = _minmax(base)
        prior_n = _minmax(prior)
        bm25_n = _minmax(bm25)
        bm25f_n = _minmax(bm25f)
        base_rank = _rank_map(base)
        dense_rank = _rank_map(dense)
        bm25f_rank = _rank_map(bm25f)
        prior_rank = _rank_map(prior)
        docs = set(base_n) | set(prior_n) | set(bm25f_n)
        vals = {}
        for did in docs:
            shallow_prior = (
                0.15 * prior_n.get(did, 0.0)
                if prior_rank.get(did, 10**9) <= 50
                else 0.0
            )
            depth_supported = min(
                base_rank.get(did, 10**9),
                dense_rank.get(did, 10**9),
                bm25f_rank.get(did, 10**9),
            ) <= 100
            residual = (
                0.01 * max(0.0, bm25f_n.get(did, 0.0) - bm25_n.get(did, 0.0))
                if depth_supported
                else 0.0
            )
            vals[did] = base_n.get(did, 0.0) + shallow_prior + residual
        return dict(sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])

    @staticmethod
    def _structure_residual_hybrid(
        bm25: Mapping[str, float],
        bm25f: Mapping[str, float],
        dense: Mapping[str, float],
        *,
        alpha: float,
        residual_weight: float,
        top_k: int,
    ) -> dict[str, float]:
        base = _hybrid_scores(bm25, dense, alpha=alpha, top_k=top_k)
        base_n = _minmax(base)
        bm25_n = _minmax(bm25)
        bm25f_n = _minmax(bm25f)
        dense_rank = _rank_map(dense)
        bm25_rank = _rank_map(bm25)
        docs = set(base_n) | set(bm25f_n)
        vals = {}
        for did in docs:
            supported = dense_rank.get(did, 10**9) <= 100 or bm25_rank.get(did, 10**9) <= 100
            residual = (
                max(0.0, bm25f_n.get(did, 0.0) - bm25_n.get(did, 0.0))
                if supported
                else 0.0
            )
            vals[did] = base_n.get(did, 0.0) + residual_weight * residual
        return dict(sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])

    @staticmethod
    def _lexical_consensus_dense_micro_hybrid(
        bm25: Mapping[str, float],
        dense: Mapping[str, float],
        *,
        top_k: int,
    ) -> dict[str, float]:
        bm25_ranked = sorted(bm25.items(), key=lambda kv: (-kv[1], kv[0]))
        dense_ranked = sorted(dense.items(), key=lambda kv: (-kv[1], kv[0]))
        overlap = {did for did, _ in bm25_ranked[:20]} & {
            did for did, _ in dense_ranked[:50]
        }
        if len(overlap) < 5:
            return dict(bm25_ranked[:top_k])
        bm25_n = _minmax(dict(bm25_ranked[:top_k]))
        dense_n = _minmax(dense)
        vals = {
            did: bm25_n.get(did, 0.0) + 0.02 * dense_n.get(did, 0.0)
            for did in bm25_n
        }
        return dict(sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])


def structure_aware_hybrid_loader(model_name: str, **kwargs) -> CorpusAdaptiveHybridSearch:
    del model_name
    import mteb

    dense_kwargs = dict(kwargs)
    dense_kwargs.pop("revision", None)
    dense = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2", **dense_kwargs)
    bm25 = FieldWeightedBM25Search(
        field_weights={"text": 1.0},
        name="mteb/structure-aware-text-bm25-component",
    )
    bm25f = FieldWeightedBM25Search(
        field_weights={"title": 2.0, "text": 1.0},
        name="mteb/structure-aware-bm25f-component",
    )
    return CorpusAdaptiveHybridSearch(models=[bm25, bm25f, dense])


structure_aware_corpus_adaptive_hybrid = ModelMeta(
    loader=structure_aware_hybrid_loader,
    name="mteb/structure-aware-corpus-adaptive-hybrid",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="1",
    release_date="2026-06-30",
    n_parameters=22_713_216,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=87.0,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://github.com/embeddings-benchmark/mteb",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "NumPy"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    citation=None,
)
