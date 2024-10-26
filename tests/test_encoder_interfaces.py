from __future__ import annotations

from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel


def test_sentence_is_encoder():
    model = SentenceTransformer("average_word_embeddings_komninos")
    assert isinstance(model, Encoder)


def test_wrapped_sentence_is_encoder_with_query_corpus_encode():
    model = SentenceTransformer("average_word_embeddings_komninos")
    model = DRESModel(model)

    assert isinstance(model, Encoder)
