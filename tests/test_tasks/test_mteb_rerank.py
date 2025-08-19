from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import CrossEncoder

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.cache import ResultCache
from mteb.models.get_model_meta import _model_meta_from_cross_encoder
from mteb.models.model_meta import ModelMeta
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import MockRetrievalTask

logging.basicConfig(level=logging.INFO)


def test_mteb_rerank(tmp_path: Path):
    prev_model = MockNumpyEncoder()
    model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    model_meta = _model_meta_from_cross_encoder(model)
    task: AbsTaskRetrieval = mteb.get_task("SciFact")
    task.load_data()
    # create fake first stage results
    cache = ResultCache(tmp_path)

    prev_result_path = task.previous_results_path(prev_model.mteb_model_meta, cache)
    prev_result_path.parent.mkdir(parents=True, exist_ok=True)
    scifact_keys = task.dataset["default"]["test"]["queries"]["id"]
    with prev_result_path.open("w") as f:
        json.dump(
            {
                "default": {
                    "test": {
                        i: {
                            "4983": 0.1,
                            "18670": 0.9,
                            "19238": 0.01,
                        }
                        for i in scifact_keys
                    }
                }
            },
            f,
        )

    task.convert_to_reranking(prev_model.mteb_model_meta, cache)
    mteb.evaluate(
        model,
        task,
        save_retrieval_results=True,
        overwrite_strategy="always",
        cache=cache,
    )
    current_result_path = task.previous_results_path(model_meta, cache)

    # read in the results
    with current_result_path.open() as f:
        results = json.load(f)

    results = results["default"]["test"]

    results = sorted(
        results["1"].keys(),
        key=lambda x: (results["1"][x], x),
        reverse=True,
    )[:2]
    # check that only the top two results are re-orderd
    assert "19238" not in results
    assert "4983" in results
    assert "18670" in results


def test_reranker_same_ndcg1(tmp_path: Path):
    de_name = "sentence-transformers/average_word_embeddings_komninos"
    revision = "21eec43590414cb8e3a6f654857abed0483ae36e"
    de = mteb.get_model(de_name, revision=revision)
    ce = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    ce_revision = "e9ea2688951463fc2791a2ea2ddfce6762900675"
    ce.mteb_model_meta = ModelMeta(  # type: ignore
        loader=None,
        name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        languages=["eng-Latn"],
        open_weights=True,
        revision=ce_revision,
        release_date="2021-04-15",
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        public_training_code=None,
        public_training_data=None,
        reference=None,
        similarity_fn_name=None,
        use_instructions=None,
        training_datasets=None,
        framework=["Sentence Transformers", "PyTorch"],
    )
    task = MockRetrievalTask()
    cache = ResultCache(tmp_path)
    de_results = mteb.evaluate(
        de,
        task,
        save_retrieval_results=True,
        cache=cache,
        overwrite_strategy="always",
    )
    task.convert_to_reranking(de, cache)
    ce_results = mteb.evaluate(
        ce,
        task,
        cache=cache,
        overwrite_strategy="always",
    )

    assert (
        de_results[0].scores["test"][0]["ndcg_at_1"]
        == ce_results[0].scores["test"][0]["ndcg_at_1"]
    )
