import gc
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

import mteb
from mteb.models import SearchEncoderWrapper
from mteb.models.model_meta import ScoringFunction
from mteb.models.search_encoder_index import FastPlaidSearchIndex
from mteb.similarity_functions import max_sim

pytest.importorskip("fast_plaid", reason="fast-plaid is not installed")

DIM = 16
NUM_DOCS = 300
NUM_QUERIES = 5
TOP_K = 10


class _MaxSimModel:
    def __init__(
        self,
        similarity_fn_name: ScoringFunction = ScoringFunction.MAX_SIM,
        model_type: tuple[str, ...] = ("late-interaction",),
    ):
        model = mteb.get_model("mteb/baseline-random-encoder")
        meta = deepcopy(model.mteb_model_meta)
        self.mteb_model_meta = meta.model_copy(
            update={
                "similarity_fn_name": similarity_fn_name,
                "model_type": list(model_type),
            }
        )

    def encode(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


def _similarity(queries, documents):
    pad = lambda e: torch.nn.utils.rnn.pad_sequence(  # noqa: E731
        list(e), batch_first=True
    )
    return max_sim(pad(queries), pad(documents))


@pytest.fixture(scope="module")
def embeddings():
    generator = torch.Generator().manual_seed(0)
    documents = [
        torch.nn.functional.normalize(
            torch.randn(int(torch.randint(8, 24, (1,), generator=generator)), DIM),
            dim=-1,
        )
        for _ in range(NUM_DOCS)
    ]
    # Queries are noisy copies of some of the documents, so exact top-1 is well defined
    queries = [
        torch.nn.functional.normalize(
            documents[i * 7][:6] + 0.1 * torch.randn(6, DIM, generator=generator),
            dim=-1,
        )
        for i in range(NUM_QUERIES)
    ]
    return queries, documents


def _build_index(documents, tmp_path: Path | None = None) -> FastPlaidSearchIndex:
    index = FastPlaidSearchIndex(
        _MaxSimModel(),
        index_dir=tmp_path,
        device="cpu",
        n_ivf_probe=64,
        n_full_scores=NUM_DOCS,
    )
    index.add_documents(documents, [f"d{i}" for i in range(len(documents))])
    return index


def test_full_corpus_recall(embeddings):
    queries, documents = embeddings
    index = _build_index(documents)

    scores, idxs = index.search(queries, TOP_K, similarity_fn=_similarity)

    exact = _similarity(queries, documents)
    exact_top = torch.topk(exact, TOP_K, dim=1).indices.tolist()
    recall = sum(
        len(set(got) & set(want)) / TOP_K
        for got, want in zip(idxs, exact_top, strict=True)
    ) / len(queries)
    # Ranks 2-10 are near-ties on random data, so 4-bit quantization reorders them between
    # index builds (fast-plaid's index build isn't deterministic); top-1 is checked exactly below
    assert recall >= 0.6
    for q_idx in range(NUM_QUERIES):
        assert len(scores[q_idx]) == len(idxs[q_idx]) == TOP_K
        assert scores[q_idx] == sorted(scores[q_idx], reverse=True)
        # the query is a noisy copy of this document
        assert idxs[q_idx][0] == q_idx * 7
    index.clear()


def test_reranking(embeddings):
    queries, documents = embeddings
    index = _build_index(documents)
    ids = list(index.idxs)
    candidates = {
        "q0": [ids[i] for i in (0, 5, 9, 40, 100)],
        "q1": [ids[i] for i in (7, 8, 150)],
        "q2": [],
    }
    query_idx_to_id = {0: "q0", 1: "q1", 2: "q2"}

    with pytest.warns(UserWarning, match="No top-ranked documents for query q2"):
        scores, idxs = index.search(
            queries[:3],
            3,
            similarity_fn=_similarity,
            top_ranked=candidates,
            query_idx_to_id=query_idx_to_id,
        )

    assert idxs[2] == [] and scores[2] == []
    # Indices are positions within each query's candidate list; the noisy copy ranks first
    assert idxs[0][0] == 0
    assert idxs[1][0] == 0
    for q_idx, qid in enumerate(["q0", "q1"]):
        assert len(idxs[q_idx]) <= 3
        assert all(0 <= i < len(candidates[qid]) for i in idxs[q_idx])
    index.clear()


def test_clear_empties_temp_index(embeddings):
    _, documents = embeddings
    index = _build_index(documents)
    tmp_dir = index.index_dir
    assert any(tmp_dir.iterdir())

    index.clear()

    assert tmp_dir.exists() and not any(tmp_dir.iterdir())
    assert index.index is None
    assert index.idxs == []
    with pytest.raises(ValueError, match="No index built"):
        index.search([documents[0]], 1, similarity_fn=_similarity)

    # the next task reuses the same directory
    index.add_documents(documents, [f"d{i}" for i in range(len(documents))])
    assert index.index_dir == tmp_dir and any(tmp_dir.iterdir())
    index.clear()


def test_temp_index_dir_removed_with_backend():
    index = FastPlaidSearchIndex(_MaxSimModel(), device="cpu")
    tmp_dir = index.index_dir
    assert tmp_dir.exists()

    del index
    gc.collect()

    assert not tmp_dir.exists()


def test_index_dir_is_kept_on_clear(embeddings, tmp_path: Path):
    _, documents = embeddings
    index = _build_index(documents, tmp_path / "plaid")
    index.clear()
    assert (tmp_path / "plaid").exists()


def test_add_documents_twice(embeddings):
    _, documents = embeddings
    index = FastPlaidSearchIndex(_MaxSimModel(), device="cpu")
    index.add_documents(documents[:200], [f"d{i}" for i in range(200)])
    index.add_documents(documents[200:], [f"d{i}" for i in range(200, NUM_DOCS)])
    assert len(index.idxs) == NUM_DOCS
    index.clear()


@pytest.mark.parametrize(
    "similarity", [ScoringFunction.COSINE, ScoringFunction.DOT_PRODUCT]
)
def test_only_max_sim_supported(similarity: ScoringFunction):
    with pytest.raises(ValueError, match="only supports"):
        FastPlaidSearchIndex(_MaxSimModel(similarity))


def test_wrapper_saves_plaid_args_as_experiment():
    model = _MaxSimModel()
    index = FastPlaidSearchIndex(
        model, nbits=2, kmeans_niters=3, n_ivf_probe=16, n_full_scores=512, seed=7
    )
    wrapper = SearchEncoderWrapper(model, index_backend=index)

    assert wrapper.mteb_model_meta.experiment_kwargs == {
        "index_backend": "fast_plaid",
        "plaid_nbits": 2,
        "plaid_kmeans_niters": 3,
        "plaid_n_ivf_probe": 16,
        "plaid_n_full_scores": 512,
        "plaid_seed": 7,
    }
    # results are saved under the experiment name, apart from exact MaxSim results
    assert wrapper.mteb_model_meta.experiment_name is not None
    assert "plaid_nbits_2" in wrapper.mteb_model_meta.experiment_name
    # the wrapped model's own meta is left untouched
    assert model.mteb_model_meta.experiment_kwargs is None


def test_wrapper_rejects_non_late_interaction_model():
    model = _MaxSimModel(model_type=("dense",))
    with pytest.raises(
        ValueError, match="FastPlaidSearchIndex supports models of type"
    ):
        SearchEncoderWrapper(model, index_backend=FastPlaidSearchIndex(model))
