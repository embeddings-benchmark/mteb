"""test that mteb.evaluate integrates with SentenceTransformers"""

import logging

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb
from mteb.abstasks import AbsTask
from mteb.models import ModelMeta
from tests.mock_tasks import (
    MockInstructionReranking,
    MockRerankingTask,
)
from tests.task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize("model_name", ["average_word_embeddings_levy_dependency"])
def test_sentence_transformer_integration(task: AbsTask, model_name: str):
    """Test that a task can be fetched and run"""
    model = SentenceTransformer(model_name)
    # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
    # SentenceTransformerWrapper would set the model's prompts to None because
    # the mock tasks are not in the MTEB task registry. The linked PR changes
    # this behavior and keeps the prompts as configured by the model, so this
    # test now sets the prompts to an empty dict explicitly to preserve the legacy
    # behavior and focus the test on the tasks instead of the prompts.
    # Using empty dict instead of None to avoid TypeError in SentenceTransformers 5.0.0+
    model.prompts = {}
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockInstructionReranking(),
    ],
)
@pytest.mark.parametrize("model_name", ["cross-encoder/ms-marco-TinyBERT-L2-v2"])
def test_sentence_transformer_integration_cross_encoder(task: AbsTask, model_name: str):
    """Test that a task can be fetched and run"""
    model = CrossEncoder(model_name)
    mteb.evaluate(model, task, cache=None)


def test_model_meta_load_sentence_transformer_metadata_from_model():
    # used also in test CLI
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    meta = ModelMeta.from_sentence_transformer_model(model)

    assert meta.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.revision is not None
    assert meta.max_tokens == 256
    assert meta.embed_dim == 384
    assert meta.similarity_fn_name.value == "cosine"


@pytest.mark.parametrize("as_sentence_transformer", [True, False])
@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_model_meta_sentence_transformer_from_hub(
    as_sentence_transformer: bool, model_name: str
):
    if as_sentence_transformer:
        meta = ModelMeta.from_hub(model_name)
    else:
        meta = ModelMeta._from_hub(model_name)

    assert meta.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.revision is not None
    assert meta.release_date == "2021-08-30"
    assert meta.n_parameters == 22713728
    assert meta.memory_usage_mb == 87
    assert meta.embed_dim == 384
    assert meta.license == "apache-2.0"
    # model have max_position_embeddings 512, but in sentence_bert_config 256
    if as_sentence_transformer:
        assert meta.similarity_fn_name.value == "cosine"
        assert meta.max_tokens == 256
    else:
        assert meta.max_tokens == 512


@pytest.mark.parametrize("model_name", ["cross-encoder/ms-marco-TinyBERT-L2-v2"])
def test_cross_encoder_model_meta(model_name: str):
    model = CrossEncoder(model_name)
    meta = ModelMeta.from_cross_encoder(model)

    assert meta.name == "cross-encoder/ms-marco-TinyBERT-L2-v2"
    assert meta.revision is not None
