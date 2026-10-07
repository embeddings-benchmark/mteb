"""tests/test_models/test_spectral_model.py - Test suite for JMullings/spectral-embed-v1-140m."""

import numpy as np
import pytest

import mteb
from mteb.models.model_implementations.spectral_models import (
    SpectralEmbedModel,
    spectral_embed_v1_140m,
)

MODEL_ID = "JMullings/spectral-embed-v1-140m"


@pytest.fixture(scope="session")
def model():
    """Instantiate model once across the test session."""
    return mteb.get_model(MODEL_ID)


def test_spectral_model_metadata():
    """Verify metadata fields and registration."""
    meta = spectral_embed_v1_140m
    assert meta.name == MODEL_ID
    assert meta.embed_dim == 2048
    assert meta.languages == ["eng-Latn"]
    assert meta.n_parameters == 139_471_360
    assert meta.similarity_fn_name == "cosine"
    assert "NFCorpus" in meta.training_datasets


def test_spectral_model_loader_and_encode(model):
    """Verify loaded model, vector shapes, L2 normalization, and semantic discrimination."""
    assert isinstance(model, SpectralEmbedModel)
    texts = [
        "How do I reset my account password?",
        "Steps to change login credentials and recover access.",
        "The weather forecast calls for rain tomorrow.",
    ]
    emb = model.encode(texts)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (3, 2048)

    # Check unit L2 norm
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Embeddings not unit norm: {norms}"

    # Check semantic discrimination: sim(P) > sim(Unrelated)
    sim_paraphrase = float(emb[0] @ emb[1])
    sim_unrelated = float(emb[0] @ emb[2])
    assert sim_paraphrase > sim_unrelated, "Semantic ordering check failed!"


def test_spectral_model_retrieval_interfaces(model):
    """Verify query and corpus interfaces."""
    q_emb = model.encode_queries(["Search query"])
    c_emb = model.encode_corpus(
        [{"title": "Doc Title", "text": "Document body content."}]
    )
    assert q_emb.shape == (1, 2048)
    assert c_emb.shape == (1, 2048)


def test_spectral_model_end_to_end_eval(model):
    """Run an end-to-end evaluation smoke test on STS12."""
    tasks = mteb.get_tasks(tasks=["STS12"])
    results = mteb.evaluate(model, tasks)
    assert len(results) > 0
