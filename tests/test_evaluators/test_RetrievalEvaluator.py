import logging
from typing import Any

import pytest
import torch
from datasets import Dataset

from mteb._evaluators import RetrievalEvaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks.retrieval import general_args
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.models.sentence_transformer_wrapper import MultiVectorSearchEncoderWrapper
from mteb.similarity_functions import cos_sim, max_sim
from mteb.timing import TimingStack

TOL = 0.0001


class TestRetrievalEvaluator:
    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalTask",
        main_score="ndcg_at_10",
        **general_args,
    )

    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """

        self.evaluator = RetrievalEvaluator(
            corpus=None,
            queries=None,
            task_metadata=self.metadata,
            hf_split=None,
            hf_subset=None,
            instructions=None,
            top_ranked=None,
            qid=None,
            top_k=5,
            timer=TimingStack(),
        )

    @pytest.mark.parametrize(
        ("relevant_docs", "results", "ignore_identical_ids", "expected_metrics"),
        [
            (
                # Qid: {Docid: Relevance}
                {
                    "0": {"0": 1, "1": 1},
                    "1": {"1": 1},
                },
                {
                    "0": {"0": 1.0, "1": 0.9, "2": 0.8},
                    "1": {"0": 0.0, "1": 1.0, "2": 0.0},
                },
                False,
                {
                    "ndcg": {"NDCG@1": 1.0, "NDCG@2": 1.0, "NDCG@3": 1.0},
                    "map": {"MAP@1": 0.75, "MAP@2": 1.0, "MAP@3": 1.0},
                    "recall": {"Recall@1": 0.75, "Recall@2": 1.0, "Recall@3": 1.0},
                    "precision": {"P@1": 1.0, "P@2": 0.75, "P@3": 0.5},
                    "task_specific": {},
                },
            ),
            # Test no self retrieval
            (
                # Qid: {Docid: Relevance}
                {
                    "0": {"0": 1, "1": 1},
                    "1": {"1": 1},
                },
                {
                    "0": {"0": 1.0, "1": 0.9, "2": 0.8},
                    "1": {"0": 0.0, "1": 1.0, "2": 0.0},
                },
                True,
                {
                    "ndcg": {"NDCG@1": 0.5, "NDCG@2": 0.30657, "NDCG@3": 0.30657},
                    "map": {"MAP@1": 0.25, "MAP@2": 0.25, "MAP@3": 0.25},
                    "recall": {"Recall@1": 0.25, "Recall@2": 0.25, "Recall@3": 0.25},
                    "precision": {"P@1": 0.5, "P@2": 0.25, "P@3": 0.16667},
                    "task_specific": {},
                },
            ),
        ],
    )
    def test_metrics_at_k(
        self, relevant_docs, results, ignore_identical_ids, expected_metrics
    ):
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            hit_rate,
        ) = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
            ignore_identical_ids=ignore_identical_ids,
        )

        assert ndcg == expected_metrics["ndcg"]
        assert _map == expected_metrics["map"]
        assert recall == expected_metrics["recall"]
        assert precision == expected_metrics["precision"]

    @pytest.mark.parametrize(
        ("ignore_identical_ids", "expected_naucs"),
        [
            (
                True,
                {
                    "nAUC_NDCG@3_max": 0.50843,
                    "nAUC_NDCG@3_std": 0.18322,
                    "nAUC_NDCG@3_diff1": 0.21416,
                },
            ),
            (
                False,
                {
                    "nAUC_NDCG@3_max": 0.8368244286523474,
                    "nAUC_NDCG@3_std": 0.9125701917627439,
                    "nAUC_NDCG@3_diff1": 0.950708977119359,
                },
            ),
        ],
    )
    def test_n_auc(self, ignore_identical_ids, expected_naucs):
        relevant_docs = {
            "0": {"0": 1, "1": 1},
            "1": {"0": 1},
            "2": {"0": 1, "1": 1, "2": 1},
            "3": {"0": 1},
            "4": {"0": 1, "1": 1},
        }
        results = {
            "0": {"0": 0.8, "1": 0.3, "2": 0.4},
            "1": {"0": 0.5, "1": 0.8, "2": 0.4},
            "2": {"0": 0.9, "1": 0.3, "2": 0.3},
            "3": {"0": 0.1, "1": 0.2, "2": 0.2},
            "4": {"0": 0.5, "1": 0.4, "2": 0.5},
        }

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            hit_rate,
        ) = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
            ignore_identical_ids=ignore_identical_ids,
        )
        aucs = ["nAUC_NDCG@3_max", "nAUC_NDCG@3_std", "nAUC_NDCG@3_diff1"]
        for auc in aucs:
            assert naucs[auc] == pytest.approx(expected_naucs[auc], TOL)


METADATA = TaskMetadata(
    type="Retrieval",
    name="MockRetrievalTask",
    main_score="ndcg_at_10",
    **general_args,
)
CORPUS = Dataset.from_dict(
    {"id": [f"d{i}" for i in range(10)], "text": [f"document {i}" for i in range(10)]}
)
QUERIES = Dataset.from_dict(
    {"id": [f"q{i}" for i in range(3)], "text": [f"query {i}" for i in range(3)]}
)
TOP_RANKED = {f"q{i}": ["d0", "d1", "d2"] for i in range(3)}
ENCODE_KWARGS = {"batch_size": 4, "show_progress_bar": False}


class _DenseEncoder:
    mteb_model_meta = None

    def encode(self, inputs, **kwargs: Any):
        return torch.randn(sum(len(batch["text"]) for batch in inputs), 8)

    def similarity(self, embeddings1, embeddings2):
        return cos_sim(embeddings1, embeddings2)


class _MultiVectorModel:
    def similarity(self, embeddings1, embeddings2):
        pad = lambda embeddings: torch.nn.utils.rnn.pad_sequence(  # noqa: E731
            list(embeddings), batch_first=True
        )
        return max_sim(pad(embeddings1), pad(embeddings2))


class _MultiVectorWrapper(MultiVectorSearchEncoderWrapper):
    corpus_chunk_size = 4

    def __init__(self):
        self.model = _MultiVectorModel()

    def _encode(self, inputs, **kwargs: Any):
        return [torch.randn(3, 8) for batch in inputs for _ in batch["text"]]

    def similarity(self, embeddings1, embeddings2):
        return self.model.similarity(embeddings1, embeddings2)


def _run_retrieval(search_model, top_ranked=None) -> TimingStack:
    timer = TimingStack()
    evaluator = RetrievalEvaluator(
        corpus=CORPUS,
        queries=QUERIES,
        task_metadata=METADATA,
        hf_split="test",
        hf_subset="default",
        top_ranked=top_ranked,
        top_k=2,
        timer=timer,
    )
    evaluator(search_model, encode_kwargs=ENCODE_KWARGS)
    return timer


@pytest.mark.parametrize(
    ("search_model", "top_ranked", "expected_phases"),
    [
        (
            SearchEncoderWrapper(_DenseEncoder(), corpus_chunk_size=4),
            None,
            ["Encoding queries", "Searching corpus"],
        ),
        (
            SearchEncoderWrapper(_DenseEncoder(), corpus_chunk_size=4),
            TOP_RANKED,
            ["Encoding queries", "Encoding corpus", "Computing similarity"],
        ),
        (
            _MultiVectorWrapper(),
            None,
            ["Encoding queries", "Searching corpus"],
        ),
        (
            _MultiVectorWrapper(),
            TOP_RANKED,
            ["Encoding queries", "Encoding corpus", "Computing similarity"],
        ),
    ],
    ids=["dense-search", "dense-rerank", "multi-vector-search", "multi-vector-rerank"],
)
def test_search_phases(search_model, top_ranked, expected_phases):
    timer = _run_retrieval(search_model, top_ranked)

    assert [phase["name"] for phase in timer.phases] == expected_phases
    assert all(
        (phase["split"], phase["subset"]) == ("test", "default")
        for phase in timer.phases
    )


def test_full_corpus_search_logs_chunk_timings(caplog):
    # 10 documents in chunks of 4 -> 3 chunks, recorded as a single phase
    with caplog.at_level(logging.INFO, logger="mteb.models.search_wrappers"):
        timer = _run_retrieval(
            SearchEncoderWrapper(_DenseEncoder(), corpus_chunk_size=4)
        )

    assert [phase["name"] for phase in timer.phases].count("Searching corpus") == 1
    assert sum("Corpus chunk" in message for message in caplog.messages) == 3
    assert any("Searched 3 corpus chunk(s)" in message for message in caplog.messages)


def test_reused_wrapper_records_to_each_tasks_timer():
    wrapper = SearchEncoderWrapper(_DenseEncoder(), corpus_chunk_size=4)
    first_timer = _run_retrieval(wrapper)
    first_phases = list(first_timer.phases)

    second_timer = _run_retrieval(wrapper)

    assert first_timer.phases == first_phases
    assert [phase["name"] for phase in second_timer.phases] == [
        "Encoding queries",
        "Searching corpus",
    ]
