import json
import logging
from pathlib import Path

from sentence_transformers import CrossEncoder

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.cache import ResultCache
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper
from tests.mock_models import MockNumpyEncoder
from tests.mock_tasks import MockRetrievalTask

logging.basicConfig(level=logging.INFO)


def test_convert_reranking(tmp_path: Path):
    model = CrossEncoderWrapper("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    task: AbsTaskRetrieval = mteb.get_task("SciFact")
    task.load_data()

    prev_result_path = tmp_path / "prev_results.json"
    scifact_keys = task.dataset["default"]["test"]["queries"]["id"]
    with prev_result_path.open("w") as f:
        json.dump(
            {
                "mteb_model_meta": MockNumpyEncoder.mteb_model_meta.to_dict(),
                "default": {
                    "test": {
                        i: {
                            "4983": 0.1,
                            "18670": 0.9,
                            "19238": 0.01,
                        }
                        for i in scifact_keys
                    }
                },
            },
            f,
        )

    task = task.convert_to_reranking(prev_result_path)
    mteb.evaluate(
        model,
        task,
        overwrite_strategy="always",
        prediction_folder=tmp_path,
    )

    # read in the results
    with (tmp_path / task.prediction_file_name).open() as f:
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
    # TODO: Not sure what this is testing exactly

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
    results_folder = tmp_path / "retrieve_results" / "de"
    de_results = mteb.evaluate(
        de,
        task,
        overwrite_strategy="always",
        prediction_folder=results_folder,
    )
    task = task.convert_to_reranking(results_folder / task.prediction_file_name)
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
